# Trend-Scanning Labels at Scale (López de Prado, ML4AM Ch.5)

Reproduces López de Prado's **trend-scanning** labeller — for each observation,
the sign of the most statistically significant local forward trend, scored by the
**t-value of an OLS slope** maximised over a band of look-forward horizons — then
tests across **Crypto + US Equities + Forex (42 instruments)** whether trend-scan
labels make a **better supervised target** for a secondary side-prediction model
than the **fixed-horizon** labeller LdP contrasts it against, and (deepening) than
a **triple-barrier** meta-overlay. The verdict is the program **headline metric,
the Deflated Sharpe Ratio (DSR)**, net of **realistic** costs, with **PBO** and
**effective-N** as supporting deflation diagnostics.

Everything runs on **real 1-minute data** (no synthetic series); the forward
look-ahead lives **only in the label** (features and the acted-on side use no
window-`t`-or-later information); the secondary model is trained/scored under
**purged k-fold CV** with embargo; equity and FX frictions come from `lib/realism`
(time-of-day half-spread + commission, **no clamping**); crypto keeps the flat
house cost.

Headline up front, because the house rule is to say so plainly:
**trend-scanning is a *modest, real* improvement on profit factor (it beats the
fixed-horizon target in 28/42 instruments, sign-test p=0.022), but the advantage
is *not* significant once you deflate — it wins DSR in only 21/42 (p=0.56), and
the count of instruments clearing DSR>0.95 is essentially tied (trend-scan 19,
fixed-horizon 18, triple-barrier 19 of 42).** Where it survives deflation, the
edge is a property of the *market and the look-forward band*, not of the labeller:
short (5,30)-bar horizons collapse everywhere, while (10,60)/(20,120) hold. The
triple-barrier meta-overlay does **not** add deflated value here and is
*significantly worse* on DSR (p=0.9995). Details below.

---

## 1. What LdP proposed (formulae)

### Trend-scanning (ML4AM §5.5)
For each observation `t`, regress **log-price** on a time index over **every**
forward window of length `L ∈ [Lmin, Lmax]`:

    y_j = a + b·j ,   j = 0..L−1 ,   y_j = logprice[t+j]

compute the OLS slope `b` and its **t-value** `t_b = b / se(b)`, with
`se(b) = sqrt( SSE/(L−2) / Sxx )` and `Sxx = Σ(x_j − x̄)²`. The **label** for `t`
is the **sign of `t_b` at the horizon `L*` that maximises `|t_b|`** (the most
statistically significant local trend); **`|t_b(L*)|` is the confidence /
meta-weight**, and `L*` is the data-chosen horizon. LdP's argument: a fixed
holding horizon is an arbitrary modelling choice that throws away information; let
the data pick the horizon at which the trend is clearest, and carry the
significance as a sample weight.

### Fixed-horizon control (what LdP contrasts against)
The same OLS-t machinery at a **single fixed `L`** for every observation — no
horizon search. This is the labeller trend-scanning is meant to beat.

### Triple-barrier meta-overlay (deepening; AFML §3.6 / ML4AM Ch.5)
The deepening adds a third *target*: with the trend-scan label as the structural
**side**, place vol-scaled profit-take / stop-loss / vertical barriers, label by
first touch on **full intrabar OHLC**, and let a secondary classifier predict
**P(the sided bet survives the barriers profitably)**. This meta-label can **veto
/ size** but never flip the side. It answers: *does a triple-barrier filter
improve the trend-scan side over trading it raw?*

### Deflated Sharpe Ratio — the headline (LdP & Bailey 2014)
A backtest selected from many trials has an inflated Sharpe. DSR is the
Probabilistic Sharpe Ratio of the *selected* strategy benchmarked against the
**expected maximum Sharpe of N skill-less trials**,
`SR₀ ≈ √(V[SR]) · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`,
with N and V[SR] taken from the trial set. We treat the **9-point IS-tunable grid
(3 horizon bands × 3 confidence quantiles) as the trials**, separately for each
target. A "win" must survive this deflation.

---

## 2. Method & data

| Market | Source (real 1m) | Bars | Cost / side |
|---|---|---|---|
| Crypto (27 perps) | `ldp_cache_1m/*_1m.parquet` | **dollar bars** (~20k) | 7 bp (flat house) |
| US Equities (7 ETFs) | Algoseek `etf_1min/*.csv.gz`, RTH only | **dollar bars** (~20k) | `lib/realism`: time-of-day half-spread + commission |
| Forex (8 pairs) | `ldp_cache_fx/*_fx1m.parquet` (no volume) | **tick bars** (~20k) | `lib/realism`: UTC time-of-day half-spread |

- **The experiment is leakage-free by construction.** You cannot trade a
  forward-looking label directly — that leaks the future into P&L. Instead, at
  each event a **bagged-tree secondary model** (`BaggingClassifier(
  DecisionTree(max_depth=4, min_leaf=20), n_estimators=40)`, LdP's recommendation)
  predicts the label sign from **causal features only**, trained & scored under
  `purged_kfold_splits` (6 folds, 1% embargo, label window purged around each test
  fold). The strategy trades the **out-of-fold predicted side**, entering at the
  event close and exiting after a **causal forward hold** of `L_fixed` bars
  (decoupled from the label's lookahead-optimal window, so trade P&L carries no
  label-endpoint selection bias).
- **Events** = bars whose `|t_b|` clears an IS-tuned confidence quantile (LdP
  samples on significant trends, not every bar).
- **Causal features** (no lookahead): EWMA vol, MA-gap, three momentum horizons,
  Wilder RSI(14), short/long vol ratio, order-flow imbalance (crypto only; 0
  elsewhere), smoothed intrabar range.
- **Realistic costs, no clamping.** Equity and FX per-side cost is the
  time-of-day half-spread (+commission floor for equity) from `lib/realism`,
  charged on entry **and** exit and scaled by bet size; weekend/overnight gaps are
  kept as real risk. Crypto keeps the flat 7 bp house default.
- **IS-tunable, not enumerated strategies.** The structural shape (trend-scan
  label + bagged-tree side model) is one strategy. The numeric knobs — horizon
  band `(Lmin,Lmax) ∈ {(5,30),(10,60),(20,120)}` and confidence quantile
  `q ∈ {0.60,0.75,0.90}` — are the **9 trials**. We select the best-out-of-fold
  trial and deflate against the trial dispersion. (`MAX_CONCURRENT` overlap is
  handled by spreading each trade's return across its holding bars.)

---

## 3. Reproduction (the textbook pieces)

- **Kernel is correct.** The multi-horizon OLS-t scan is a Numba `@njit` kernel
  with **incremental** running sums (O(1) per extra horizon bar). An independent
  NumPy reference (full refit per horizon) verifies it: **label & `L*`
  bit-identical, realised window return bit-identical** (`max|Δ|=0`); the t-value
  itself differs by `~1e-6` (float summation-order only — signs, argmax horizon,
  and returns that drive selection and P&L are exact). The triple-barrier kernel
  reused from project 03 is likewise verified bit-identical on the trend-scan
  event set (`touch/label/hold` identical, `max|Δ ret_gross| = 8.9e-16`).
- **The data-chosen horizon is real.** Median winning horizon `L*` ranges
  24–111 bars across instruments and tracks the band: short bands pick short `L*`,
  wide bands pick long `L*`. Trend-scanning is genuinely selecting *where* the
  trend is clearest, not collapsing to a corner.
- **Confidence is informative as a weight, not as a gate.** `|t_b|` at events
  spans a wide distribution (`fig2_tval_confidence.png`); sizing by the secondary
  model's confidence is what turns the signed label into a tradeable size.

---

## 4. At-scale multi-market result

### Trend-scan vs fixed-horizon (the core question)

By-market medians (net of realistic costs; SR annualised; DSR is the headline):

(deepening run; med values across the market's instruments):

| market | n | med PF fix→TS | med SR_ann (TS) | med DSR fix→TS | #DSR>0.95 fix / TS | med PBO (TS) |
|---|---|---|---|---|---|---|
| crypto   | 27 | 1.22 → **1.36** | 4.88 | 0.508 → 0.376 | 11 / **12** | 0.46 |
| equities | 7  | 1.86 → **1.89** | 4.03 | 0.004 → **0.340** | 3 / 3 | 0.06 |
| forex    | 8  | 1.50 → 1.46 | 7.99 | 0.753 → **0.853** | 4 / 4 | 0.47 |

Cross-sectional tallies (42 instruments, deepening run):

- **PF: trend-scan beats fixed-horizon in 28/42 (sign-test p=0.022)** — a real,
  repeatable, if modest, improvement on profit factor.
- **DSR: trend-scan wins 21/42 (p=0.56)** — directionally favourable but **not
  significant** once deflated. The median-DSR gaps in the table point in
  conflicting directions across markets (fixed-horizon higher in crypto,
  trend-scan higher in equities) — a symptom of DSR saturation (see §5), which is
  why the survivor count and the sign test, not the median, are the headline.
- **DSR>0.95 survivors: 19 (trend-scan) vs 18 (fixed-horizon)** — a one-instrument
  edge, i.e. essentially tied on the metric that actually counts. (The core 2-target
  run lands at 19 vs 15; the 15→18 fixed-horizon wobble between runs is itself the
  saturation fragility in action — a few crypto/forex names sit on the 0/1 DSR
  boundary and flip under tiny numerical differences across worker processes.)
- `fig1_pf_dsr_by_market.png` (left) shows trend-scan PF ≥ fixed-horizon in every
  market; the representative equity curves (`fig3_equity_curves.png`) are honest —
  on BTC and EURUSD the fixed-horizon line is actually slightly *ahead*, so the
  aggregate edge is not universal and the panels aren't cherry-picked winners.

### Deepening 1 — triple-barrier meta-overlay (does a TBM filter help?)

`fig5_three_target_dsr.png`, `tables/deepen_by_market.csv`:

- **No.** Triple-barrier-meta beats trend-scan on PF in only 19/42 (p=0.78) and on
  **DSR in only 11/42 (p=0.9995 — i.e. significantly *worse*)**. DSR>0.95
  survivors are identical (crypto 12, equities 3, forex 4).
- **Why:** with symmetric 2σ barriers and a `L_fixed`-bar vertical, the vertical
  almost never binds (median timeout rate ≈ 0.06%) and the meta-model **acts ~97%
  of the time** — it barely filters. It mostly adds an over-confident, near-binary
  sizing that *increases* trial dispersion and so *raises* the deflation hurdle
  without adding edge. A triple-barrier overlay only helps when the barriers
  actually create a selective veto; on these always-touched barriers it is dead
  weight.

### Deepening 2 — robustness across the look-forward band

`fig6_band_robustness.png`, `tables/deepen_band_robustness.csv` (trend-scan DSR,
deflated within each band over its 3 quantile trials):

| market | (5,30) | (10,60) | (20,120) |
|---|---|---|---|
| crypto   | 0.497 (PF 1.06) | 0.833 (PF 1.20) | **0.882 (PF 1.23)** |
| equities | **0.000 (PF 0.79)** | 0.981 (PF 1.20) | 1.000 (PF 1.89) |
| forex    | 0.972 (PF 1.24) | 0.997 (PF 1.26) | 1.000 (PF 1.35) |

- **The edge depends on a long-enough look-forward window, not on a hand-picked
  one.** The (10,60) and (20,120) bands are robustly strong (median DSR 0.83–1.00,
  PF>1) across all three markets; the short **(5,30) band collapses** — equities
  go to DSR 0 / PF 0.79, crypto halves. Short forward windows are dominated by
  microstructure noise and the per-turnover cost, so the OLS-t has nothing
  durable to lock onto. This is the cleanest positive finding: trend-scanning is
  worth it **at multi-hour horizons**, and the band choice matters more than the
  trend-scan-vs-fixed distinction.

---

## 5. Notes, opinions, extensions (honest)

- **Is trend-scanning a better target? Modestly, and mostly on PF — not on
  deflated Sharpe.** The signed max-`|t|` label is a *slightly* cleaner target
  than a fixed-horizon slope (more horizon-adaptive, fewer mislabeled
  whipsaws), and that shows up as a significant PF win. But the *size* of the
  improvement is too small to move the deflation needle: the DSR>0.95 survivor
  count is 19 vs 18. The honest reading is **trend-scanning is a marginally better
  labelling convention, not an alpha source.**
- **DSR saturates to {0,1} per instrument — read the *count*, not the median.**
  Each instrument has 1,900–7,900 events, so the PSR z-statistic scales with
  `√(n_obs−1)` and the normal CDF saturates: per-instrument DSR is almost always
  0.000 or 1.000. The *median DSR* therefore overstates between-target gaps (a
  market that flips from 0 to 1 on a few names swings the median wildly). The
  robust summary is **# instruments with DSR>0.95** (and the cross-sectional sign
  test), which is why the headline leans on those. This is a genuine property of
  DSR at large `n_obs`, worth flagging for anyone applying it to high-event-count
  intrabar studies.
- **The meta-overlay's failure is instructive, not a bug.** It is the same lesson
  as project 03: a meta-filter can only concentrate edge that already exists, and
  if it isn't actually selective (here ~97% act-rate) it just inflates the trial
  dispersion and hurts deflation. Calibrating the barriers to produce a real veto
  (asymmetric `pt/sl`, tighter vertical) is the obvious follow-up, but it would be
  re-tuning, not a structural change.
- **Where it works best: forex and the wide bands.** Forex shows the highest
  absolute DSR for both labellers (tight 1-pip-class costs + persistent
  macro-driven trends); the (20,120) band is strongest everywhere. Crypto shows
  the *largest TS-over-fix lift* but also the highest PBO (~0.46), so its best-IS
  trial often doesn't stay best OOS — consistent with the modest, not decisive,
  edge.
- **Extensions that would actually matter** (none of which are "tune harder"):
  (1) **sample weighting by label uniqueness / return attribution** (AFML Ch.4) so
  overlapping trend-scan windows don't over-count — likely the single biggest
  honesty improvement, since events here overlap heavily; (2) **CPCV** instead of
  single-path k-fold for a *distribution* of OOS DSRs and a tighter PBO;
  (3) a **minimum-acted-trades floor** in trial selection; (4) **asymmetric /
  selective barriers** so the TBM overlay actually vetoes; (5) feature MDA on the
  purged folds to prune noise features.

---

## 6. Performance (profile → Numba → bit-identical check)

- **Profiling finding.** As in project 03, cProfile of a single instrument shows
  the run is **dominated by the sklearn bagged-tree CV fits, not the trend-scan
  kernel.** LdP's prose flags the per-observation multi-horizon refit as the hot
  loop; we pre-empted it with the incremental Numba kernel, which pushed the scan
  below the noise floor and left the ML where the cost should be.
- **Numba kernel.** The incremental-sums `_trend_scan_kernel` runs **≈1176×**
  faster than the pure-NumPy full-refit reference (n=3000, band (20,120):
  4502 ms → 3.8 ms). The deepening adds the project-03 triple-barrier kernel
  (single forward pass per event).
- **Bit-identical checks** (`--verify`): trend-scan label/`L*`/return exact vs the
  independent reference; triple-barrier `touch/label/hold` exact, `ret_gross`
  agrees to `8.9e-16`. Both kernels are trustworthy; the float-epsilon t-value /
  return deltas never touch a sign or an argmax.
- **At-scale wall-clock.** Core run (2 targets, 42 instruments, 9 trials): ~20
  min single-process. Deepening run (3 targets incl. TBM, 6-way parallel): **440 s
  / 42 instruments**, peak RAM ~18 GB across 6 workers (well under the 46 GB box).

---

## 7. Limitations & exact rerun commands

- **DSR saturation at high `n_obs`** (see §5): per-instrument DSR is near-binary;
  we report survivor counts + sign tests, not the median DSR, as the headline.
- **Single-path CV.** Purged k-fold (one OOS path). CPCV would give a distribution
  of OOS DSRs and a tighter PBO; deferred to keep the run short.
- **Overlapping events are not uniqueness-weighted.** Trend-scan windows overlap
  heavily; AFML-Ch.4 sample weights would be the principled next step and would
  likely shrink some of the strong-looking DSRs.
- **Crypto cost is flat house (7 bp)**; only equity/FX use the time-of-day
  `lib/realism` schedule. A per-bar crypto spread feed would tighten the crypto
  numbers.
- **Bar counts (~20k)** are set for cross-market comparability, not maximal
  resolution on the 19-year equity histories.
- **Reproduce:**
  ```bash
  cd /home/daru/ldp_review/projects/05_trend_scanning
  python3 scripts/run_trend_scanning.py --verify    # trend-scan kernel vs reference
  python3 scripts/run_trend_scanning.py             # core 2-target run (~20 min)
  python3 scripts/deepen_labels.py --verify         # TBM kernel vs reference on TS events
  python3 scripts/deepen_labels.py --jobs 6         # 3-target + band robustness (~7-8 min)
  ```
  Core outputs: `tables/{per_instrument,by_market_summary}.csv`,
  `tables/raw_results.parquet`, `figures/fig{1..4}_*.png`.
  Deepening outputs: `tables/deepen_{per_instrument,by_market,band_robustness}.csv`,
  `tables/deepen_results.md`, `tables/deepen_raw.parquet`,
  `figures/fig5_three_target_dsr.png`, `figures/fig6_band_robustness.png`.

---

## 8. Paper-worthiness

**Honest verdict: paper-worthy as a careful *methodological / comparative* result,
not as a "trend-scanning makes money" result.** The contribution is a clean,
multi-market (crypto + equities + forex), costed, leakage-controlled, DSR-gated
head-to-head of three labelling targets feeding one identical secondary model,
with two findings the literature tends to gloss:

1. **Trend-scanning is a *marginally* better target — significant on profit factor
   (28/42, p=0.022) but not on deflated Sharpe (DSR>0.95 survivors 19 vs 18).**
   That gap between "improves raw PF" and "survives deflation" is exactly the
   distinction practitioners under-report, and showing it across three asset
   classes with a shared harness is the useful part.
2. **A triple-barrier meta-overlay adds *no* deflated value here and is
   significantly worse on DSR**, because its barriers weren't selective — a
   concrete, reproducible cautionary case for "add a meta-filter" reflexes.
3. **The robust, transferable effect is the look-forward band, not the labeller:**
   short (5,30)-bar horizons collapse across all markets while (10,60)/(20,120)
   hold — a clean, defensible robustness curve.

It also surfaces a genuinely useful technical note: **DSR saturates to {0,1} at
the high event counts typical of intrabar studies, so the survivor-count / sign
test is the honest summary, not the median DSR.** To upgrade to a *positive*
paper you would add AFML-Ch.4 uniqueness weighting (to discount the overlapping
trend-scan windows) and CPCV (for a DSR distribution), and likely conclude with a
sharper, more honest version of the same modest edge. The infrastructure here
(verified Numba trend-scan + triple-barrier kernels, purged CV, realistic costs,
DSR/PBO/effective-N harness) is exactly what that study would reuse.
