# Project 13, Portfolio Construction: Denoising/Detoning, HRP, NCO, TIC vs the Markowitz Curse

**Walk-forward, real multi-market data. Does López de Prado's portfolio toolkit actually
beat Markowitz out-of-sample, here?**

---

## 1. Question & scope

López de Prado argues that mean-variance optimisation on a *sample* covariance is a trap
(the "Markowitz curse": the more correlated the assets, the more unstable and concentrated
the optimal weights). His proposed fixes are (a) **covariance denoising/detoning** via the
Marčenko-Pastur (MP) law, (b) **Hierarchical Risk Parity (HRP)**, clustering +
quasi-diagonalisation + recursive bisection, *no matrix inversion* (AFML Ch.16), (c)
**Nested Clustered Optimization (NCO)**, optimise within clusters then across them
(ML4AM Ch.7), and (d) **Theory-Implied Correlation (TIC)**, shrink the sample correlation
toward a taxonomy-implied block structure (LdP & Lewis 2018).

We reproduce all four and benchmark them **walk-forward** against the raw-covariance
controls (min-variance, mean-variance/tangency), inverse-variance, and naive 1/N, on
**real, multi-market** data, asking the only question that matters: *do the LdP methods
lower realised out-of-sample portfolio variance, and survive a deflated-Sharpe (DSR)
multiple-testing correction, more than the controls?*

**Two universes:**
1. **Across assets**, daily close-to-close returns of **44 instruments**: crypto perps
   (1m→daily), US equity ETFs (Algoseek 1-min→daily), FX majors (1m→daily).
2. **Across strategies** (the LdP use-case), up to **300 greedily de-correlated
   per-strategy daily-PnL series per market**, across **39 markets**, allocating a risk
   budget across them.

## 2. Method (house rules, all enforced in code)

- **Real data only**, multi-market (crypto + equities + FX). No synthetic data.
- **Walk-forward, causal**: weights estimated on a 252-day **in-sample** window, **held**
  over the following 63-day **out-of-sample** window, rolled by 63 days. Weights are
  **never** scored on the data that built them. (`walk_forward()`.)
- **Headline = realised OOS annualised volatility** (lower is better) + the **Deflated
  Sharpe Ratio** of each allocator's realised OOS stream, deflated against the 9-allocator
  menu (`lib/overfit.py::deflated_sharpe_ratio`, False Strategy Theorem).
- **Diagnostics**: concentration (HHI, effective-N of weights) and the **condition number
  of the covariance actually fed to the allocator**, the direct measure of the curse.
- **Vol-target display**: the raw strategy PnL is in *dollar* units (per-strategy daily std
  ≈ \$55), which makes cross-leg risk and Sharpe magnitudes meaningless. We add a causal,
  IS-only per-leg vol scaler (`--vol-target 0.10`) so every leg enters at a common 10%
  annualised risk; the allocator *ranking* is scale-invariant, but Sharpe becomes
  interpretable. (Under vol-targeting `inv_var ≡ 1/N` by construction, a consistency check
  the run reproduces exactly.)
- **Numba HRP kernel** verified **bit-identical** to the numpy reference (max |Δw| ≤ 5.6e-17)
  on N ∈ {8,17,40,120} every run (`--selftest`, gate in `run_full.sh`).

## 3. Headline results

### 3.1 Assets universe (44 instruments, real returns)

OOS annualised vol (raw return units, `tables/portfolio_assets_full.csv`), best→worst:

| allocator            | OOS vol | OOS Sharpe | DSR   | eff-N | cond. number |
|----------------------|--------:|-----------:|------:|------:|-------------:|
| tic_nco              | 0.146   | 0.10       | 0.34  | 3.8   | 1.5e4        |
| **hrp**              | 0.159   | 0.15       | 0.43  | 4.9   | 1.9e16       |
| hrp_denoise          | 0.162   | 0.16       | 0.44  | 4.8   | 3.6e4        |
| inv_var              | 0.167   | 0.27       | **0.64** | 6.8 | 1.9e16     |
| nco_denoise_detone   | 0.180   | 0.34       | **0.75** | 6.3 | 1.7e16     |
| nco                  | 0.181   | −0.04      | 0.15  | 2.5   | 1.9e16       |
| 1/N                  | 0.282   | 0.09       | 0.32  | 18.2  | 1.9e16       |
| min_var_raw          | 0.292   | −0.03      | 0.17  | 2.1   | 1.9e16       |
| **mean_var_raw**     | 0.418   | 0.16       | 0.44  | 2.6   | 1.9e16       |

**The toolkit works on the variance objective.** TIC-NCO and HRP deliver the **lowest OOS
vol**; raw mean-variance is the **worst (0.418 vs 0.146)**, a **~65% higher** realised OOS
vol than the best LdP method, with a degenerate effective-N of 2.6 (it bets the book on a
handful of names). This is the Markowitz curse, measured. On the *risk-adjusted* axis the
best DSR is split between `inv_var` (0.64) and `nco_denoise_detone` (0.75), i.e. the
honest winner on OOS Sharpe is the parameter-free inverse-variance / NCO, not raw Markowitz.

### 3.2 Strategies universe (39 markets, ≤300 de-correlated strategies each)

This is LdP's intended use-case and the **cleanest demonstration of the curse**. The
OOS-vol *ranking* is scale-invariant, so it holds in raw units; vol-targeted Sharpes are in
`tables/portfolio_strategies_voltgt_n300.csv`.

**Raw Markowitz is last in 39/39 markets, in every framing.** We ran the universe at two
sizes against the same 252-day IS window. Two views are reported because they measure
different things, the *ranking* is robust; the *magnitude* of HRP's edge depends on how legs
are scaled.

*Raw dollar-PnL units* (`portfolio_strategies_full.csv`, the corpus as it ships): HRP beats
raw min-variance Markowitz on OOS vol in **39/39 markets** (median **~50%** lower) and
mean-variance in **39/39** (median **~55%** lower). But this gap is *inflated* by the raw
dollar magnitudes, the singular-cov Markowitz weights blow up on a handful of high-\$-vol
legs, so much of the "win" is just HRP not concentrating into a few large-notional names.

*Vol-targeted (10%/leg, comparable risk, the fair comparison)*, `...voltgt_n{300,150}.csv`:
- **N=300 (q=0.84, sample cov SINGULAR):** mean OOS-vol rank **1/N = inv_var = 2.5 (best),
  hrp = hrp_denoise = 2.9, nco 5.3, nco_denoise_detone 5.8, then raw Markowitz LAST
  (min_var 6.7, mean_var 7.3)**. HRP < min-variance in **38/39** markets (39/39 vs
  mean-variance), median **~15%** lower vol once legs are equal-risk.
- **N=150 (q=1.68, sample cov WELL-POSED):** essentially the same ordering, **1/N = inv_var
  2.3, hrp_denoise 3.3, hrp 3.6, nco 5.1-5.4, raw Markowitz last (6.8 / 7.3)**; HRP < min-var
  in **36/39**, median **~18%** lower.
- **Key nuances:** (i) raw mean/min-variance Markowitz is the worst allocator on OOS vol in
  *both* q regimes and *both* unit conventions, the curse is real and unconditional here.
  (ii) On a *comparable-risk* basis the honest winner is **plain 1/N / inverse-variance**,
  with HRP a close second; the elaborate methods (NCO, detoning) actually rank *worse* than
  1/N in this net-noisy strategy panel, a caution against over-engineering. (iii) Under
  vol-targeting **inv_var ≡ 1/N exactly** (the run reproduces identical OOS streams), the
  expected consistency check.

**Honest negative (both N):** the de-correlated strategy samples are *net-losing* (median OOS
Sharpe ≈ −4.9 at N=300, ≈ −4.3 at N=150, vol-targeted). The greedy-decorrelation sampler
deliberately pulls uncorrelated names from a corpus dominated by losers, and **no allocator
can manufacture return from a losing menu**, DSR ≈ 0 everywhere here. The portfolio methods
control **risk**, not sign. We report this rather than hide it behind a return-positive
cherry-pick.

## 4. The Markowitz curse, quantified, and the denoising fix

`tables/condition_number_deepening.csv` (median per-fold condition number of the cov fed to
the allocator):

| universe          | N    | q=T/N | cond (raw) | cond (denoised) | reduction | signal factors |
|-------------------|-----:|------:|-----------:|----------------:|----------:|---------------:|
| assets            | 9*   | 28.0  | 8.6e3      | 9.0e2           | 9.6×      | 1 of 9         |
| strategies N=300  | 300  | 0.84  | 4.3e17     | 3.5e17          | 1.2×      | 300 of 300     |
| strategies N=150  | 150  | 1.68  | 2.1e3      | 5.0e2           | 4.1×      | **7 of 150**   |

\*per-fold active-instrument count after dropping holiday-flat columns; q is computed on
that active count (the full panel is 44 instruments but FX/crypto/equity trade on different
calendars, so each 252-day fold has ~9 jointly-active names).

Two findings:

1. **Denoising only helps when q = T/N > 1.** The MP law is *undefined* for a singular
   (q<1) sample covariance, and the strategy universe at N=300 with a 252-day IS window
   has **q = 0.84 < 1**, so the sample cov is **rank-deficient with condition number
   ~1e17-1e19** and MP denoising is a **no-op** (it keeps all 300 eigenvalues as "signal").
   This is *precisely* LdP's motivation for HRP: when you have more strategies than
   in-sample days, **you cannot invert the covariance at all**, and the inversion-free
   methods (HRP, inverse-variance) are the only ones that don't blow up. We confirm it: HRP
   and inv_var dominate exactly the regime where raw Markowitz is mathematically broken.
2. **Detoning can *re-inflate* the condition number.** Removing the market eigenvector
   leaves a near-singular residual (smallest eigenvalue → 0), so `denoise_detone` shows a
   *higher* condition number than plain `denoise`. Detoning is a clustering aid, not a
   conditioning fix, a nuance worth flagging against a naive "always detone" reading.

When q>1, denoising *does* work, and the controlled comparison is decisive. Halving the
strategy menu to **N=150** (same 252-day IS window → q=1.68) flips the covariance from
singular (cond 4.3e17) to **well-posed (cond 2.1e3)**; MP denoising then cuts it **4.1× to
5.0e2** and, the LdP money-shot, finds only **7 of 150 eigenvalues are signal**, flattening
the other 143 as Marčenko-Pastur noise. The *same* method that is inert at q=0.84 becomes
materially useful at q=1.68. In the assets universe (q≈28) denoising cuts the condition
number ~9.6×, and TIC shrinks it from ~1e16 to ~1e4 (≈12 orders of magnitude) by imposing a
well-conditioned block structure. **The single most important practical takeaway of this
project: denoising's value is entirely a function of T/N, and a practitioner with N≈300
strategies and one year of daily data is in the regime where it cannot help and HRP /
inverse-variance are mandatory.**

## 5. Verdict, does HRP/NCO actually beat Markowitz OOS, as LdP claims?

**On the variance objective: yes, raw Markowitz is reliably the worst, but the honest margin
is modest once legs are equal-risk.**
- Assets: HRP/TIC-NCO deliver ~65% lower OOS vol than raw mean-variance, and ~45% lower than
  raw min-variance, with far less concentration.
- Strategies: raw Markowitz is **last in 39/39 markets** in every framing. On the fair,
  vol-targeted basis HRP beats it in **38/39** by a median **~15%**, a real but unspectacular
  edge; the un-normalised "~50%" headline is partly an artifact of raw dollar magnitudes. The
  curse is unconditional (worst in both q regimes); HRP's *advantage over the simplest robust
  baselines* is what's conditional.

**On the risk-adjusted (Sharpe/DSR) axis: the honest winner is the simplest robust method.**
- In the assets universe the best *deflated* Sharpe belongs to **inverse-variance** and
  **NCO-denoise-detone**, not the inversion-based Markowitz, and not always HRP. HRP's edge
  is *risk reduction and de-concentration*, not alpha.
- LdP's strongest specific claim, that the full denoise→detone→NCO pipeline dominates, is
  **only partially supported here**: NCO needs denoised, well-conditioned input to behave
  (raw NCO posts a negative OOS Sharpe and the worst DSR-among-NCO), and detoning trades
  conditioning for clustering quality. The robust, parameter-light methods (HRP,
  inverse-variance, denoised-NCO) are what survive OOS; the fragile, inversion-heavy methods
  (raw mean/min-variance) are what the data punishes.

So: **HRP/NCO/denoising beat raw Markowitz at controlling out-of-sample variance,
reproducibly (Markowitz is last in 39/39 strategy markets and worst in the 44-asset universe)
, but on a comparable-risk basis the edge over the simplest robust baselines (1/N,
inverse-variance) is modest (~15% in the strategy universe), and those baselines are often the
honest OOS winner; none of the methods turns a losing menu into a winning one.** That is a
faithful, unembellished confirmation of the *spirit* of LdP's work (matrix inversion on a
noisy/singular sample covariance is the enemy; structure and shrinkage help) without
overselling either the magnitude of HRP's edge or the specific NCO-pipeline ranking.

## 6. Paper-worthiness

**Moderate-to-good as a methods/replication note; not a novel-method paper.** Strengths:
(i) genuinely multi-market real data, (ii) strict walk-forward with DSR deflation,
(iii) the **q=T/N regime dependence** of denoising is a clean, teachable, quantified result
that most HRP/NCO write-ups gloss over, (iv) the detoning-vs-conditioning nuance, and (v)
the 39/39 strategy-universe sweep is a strong, honest empirical claim. Limitations for
publication: the strategy corpus is net-losing (so the Sharpe story is muted), and the
methods themselves are LdP's, not new. **Best home:** a rigorous empirical-replication
section of the broader LdP-review program, or a focused note titled *"When does covariance
denoising actually help? A walk-forward, multi-market audit of HRP/NCO/TIC vs Markowitz"*,
with the q-regime finding as the hook.

## 7. Reproduce

```bash
bash run_full.sh                                   # both universes, raw units, all markets
# vol-targeted (clean Sharpe display):
scripts/run_portfolio.py --universe assets     --vol-target 0.10 --tag assets_voltgt
scripts/run_portfolio.py --universe strategies --n-strat 300 --vol-target 0.10 --tag strategies_voltgt_n300
scripts/run_portfolio.py --universe strategies --n-strat 150 --vol-target 0.10 --tag strategies_voltgt_n150
scripts/run_portfolio.py --selftest                # Numba HRP bit-identity gate
python3 scripts/make_figures.py                    # figures from the tables
```

## 8. Files

- `scripts/run_portfolio.py`, engine (allocators, MP denoise/detone, HRP/NCO/TIC, WFO, DSR,
  Numba HRP kernel + bit-identity self-test, `--vol-target`).
- `scripts/make_figures.py`, figures from the tables (no re-run).
- `tables/portfolio_assets_full.csv`, assets, raw return units (headline §3.1).
- `tables/portfolio_assets_voltgt.csv`, assets, vol-targeted (clean Sharpe).
- `tables/portfolio_strategies_full.csv`, 39 markets, raw units (rankings §3.2).
- `tables/portfolio_strategies_voltgt_n300.csv`, 39 markets, vol-targeted (q<1, singular).
- `tables/portfolio_strategies_voltgt_n150.csv`, 39 markets, vol-targeted (q>1, denoising active).
- `tables/condition_number_deepening.csv`, the Markowitz-curse / denoising quantification (§4).
- `figures/fig{1..5}_*.png`, OOS vol, strategy ranks, condition number, DSR-Sharpe, q-regime.
- `lib/overfit.py` (read-only), DSR / False Strategy Theorem.
