# Optimal Trading Rules without Backtesting (OU) + Triple Penance

**López de Prado, *Advances in Financial Machine Learning* Ch. 13; Bailey & López de Prado, "Determining Optimal Trading Rules without Backtesting" (2014); Bailey & López de Prado, "Stop-Outs Under Serial Correlation and the Triple Penance Rule" (2015); Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).**

> **STATUS: COMPLETE.** Full 42-instrument multi-market run done (`tables/`, `figures/`),
> plus a three-arm deepening study (`tables/deepen_*`, `figures/fig5–7`). Both the OU
> Monte-Carlo mesh kernel and the OOS apply-rule kernel are Numba and verified
> bit-identical against independent pure-Python references. Every number below is from
> the real out-of-sample evaluation; the only synthetic step is the labelled OU rule
> derivation.

---

## 1. What this reproduces, and the question it answers

López de Prado's Ch. 13 derives a profit-take / stop-loss rule **without backtesting**:
fit an Ornstein–Uhlenbeck (OU) process to a mean-reverting series, Monte-Carlo many
paths off the *fitted* process, and read the optimal (profit-take, stop-loss) pair off a
Sharpe mesh. The companion "Triple Penance" paper gives the serial-correlation-aware
maximum drawdown / time-under-water of an AR(1) return stream. We implement both and ask
the only question that matters in practice:

> **Does the OU-derived rule beat a plainly IS-tuned fixed PT/SL control out-of-sample,
> judged by the Deflated Sharpe Ratio, net of realistic costs — and on which
> mean-reverting candidates does it help or fail?**

The honest answer, established below, is **no**: on a vanilla z-score mean-reversion
entry the OU optimal-rule apparatus adds nothing decision-relevant over an IS-tuned
control, and the OU mesh exhibits a structural degeneracy (it always pins the stop at the
grid edge) that we trace, correct, and show is *intrinsic* to a first-touch rule on a
mean-reverting process. The Triple Penance section, by contrast, produces a clean,
citable empirical result.

### The one sanctioned synthetic step (labelled)

The OU optimal-rule grid is a **Monte-Carlo on a *fitted* data-generating process** — the
backtest-free rule-derivation LdP designed. The OU parameters (E₀, φ, σ) are fit to a
**real, causal in-sample** z-series; the resulting (pt\*, sl\*) is then **validated on
real out-of-sample bars with costs and intrabar OHLC exits**. Synthetic paths appear
*only* inside the rule-derivation step; every reported headline is real-OOS.

---

## 2. Formulae

**OU / AR(1) fit (causal, IS-only).** On the stationary level series xₜ,

    xₜ = a + φ·x_{t−1} + εₜ,    E₀ = a/(1−φ),    half-life = −ln 2 / ln φ,    0 < φ < 1.

**OU optimal rule (Ch. 13).** Given (E₀, φ, σ), simulate N OU paths; for each
(profit-take, stop-loss) cell exit at first touch (or a horizon cap); the cell maximising
the per-path Sharpe is (pt\*, sl\*).

**Deflated Sharpe Ratio — the headline.** DSR = PSR of the selected strategy benchmarked
against the expected maximum Sharpe of N skill-less trials,

    SR₀ ≈ √V[SR]·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))).

The 27-point IS-tunable knob grid is the trial set. A "win" must clear this deflation
(we use the conventional DSR > 0.95).

**Triple Penance.** Under a Gaussian AR(1) with lag-1 autocorrelation φ, the long-run
variance inflates by **k(φ) = (1+φ)/(1−φ)** (effective σ scaled by √k). With per-bar drift
μ > 0 and vol σ, the closed-form 95%-confidence bounds are MaxDD ≈ (z·σ)²/(4μ) and
MaxTuW ≈ (z·σ/μ)²; both inflate by k under serial correlation. When μ ≤ 0 the drawdown is
unbounded (flagged, not reported).

---

## 3. Method & data (real 1-minute, costed, causal)

| Market | Source (real 1m) | Bars (~20k) | Per-side cost |
|---|---|---|---|
| Crypto (27 perps) | `ldp_cache_1m/*_1m.parquet` | dollar bars | 7 bp (flat) |
| US Equities (7 ETFs) | Algoseek `etf_1min/*.csv.gz`, RTH only | dollar bars | time-of-day half-spread + $0.0035/sh commission (`lib/realism`) |
| Forex (8 pairs) | `ldp_cache_fx/*_fx1m.parquet` (no volume) | tick bars (on `count`) | UTC time-of-day half-spread schedule (`lib/realism`) |

- **Entry (causal, vol-scaled MR).** Rolling z-score of log price; an event fires when
  |z| first crosses an entry threshold; side = −sign(z) (fade the deviation).
- **IS/OOS.** Chronological, IS = first 60% (OU fit + control tuning), OOS = last 40%.
- **OU rule.** OU fit on the IS z-series; MC mesh over pt, sl ∈ {0.25…3.0} (12×12,
  z-units), 20,000 paths × horizon 500; argmax-Sharpe cell → (pt\*, sl\*).
- **Control.** IS-tuned fixed (pt, sl): grid-search the same 12×12 thresholds on IS real
  bars by IS net Sharpe; evaluate the winner OOS. OU and control **share the entry signal
  and costs**; only threshold-selection differs.
- **Costs** charged per side on entry **and** exit (equity/forex via `lib/realism`'s
  causal time-of-day schedules; crypto flat 7 bp).
- **No leakage.** OU fit + control tuning use IS only; OOS exits use full intrabar
  high/low first-touch (adverse barrier checked first); equities use within-session RTH
  bars so overnight gaps aren't mislabeled.
- **IS-tunable, not enumerated strategies.** One structural shape (z-score MR entry × OU
  mesh); the 27 numeric-knob combinations (z-span ∈ {50,100,200}, entry|z| ∈
  {1.5,2.0,2.5}, max_hold ∈ {50,100,200}) are the **trials** fed to the DSR / PBO.

---

## 4. Performance engineering (profile → Numba → bit-identical)

- **Hot loop = the OU Monte-Carlo mesh**, exactly as LdP's prose flags. At full sizing
  (12×12 × 20,000 paths × horizon 500) one mesh runs in **≈1.07 s** with the Numba
  `_ou_mesh_kernel` vs **≈50.8 s** for the pure-Python reference — **≈48× faster**.
- **OOS exit scan** (`_apply_rule_kernel`) is also Numba (single forward pass, full OHLC
  first-touch).
- **Bit-identical checks** (`--verify`), common random stream:
  - OU mesh: max|Δ Sharpe| = 1.1e-16, max|Δ meanPnL| = 1.1e-15, argmax cell identical.
  - apply-rule: max|Δ ret_gross| = 8.9e-16, label/hold Δ = 0 (exact).
  - corrected enter-at-deviation mesh (`deepen.py --verify`): max|Δ Sharpe| = 6.9e-11
    (variance-reduction ordering), max|Δ meanPnL| = 0, argmax identical.
  Differences are at machine epsilon (FP non-associativity); the kernels are trustworthy.

---

## 5. Results — the headline (OU rule vs IS-tuned control, OOS DSR)

**Per-market medians (42 instruments, OOS, net of costs):**

| market | n | med OU PF | med ctrl PF | med OU SR(ann) | med ctrl SR | med OU DSR | med ctrl DSR | n DSR>0.95 (OU) | n DSR>0.95 (ctrl) | med half-life (bars) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| crypto | 27 | 0.780 | 0.773 | −1.05 | −1.36 | 0.0001 | 0.000 | 0 | 0 | 22.6 |
| equities | 7 | 0.577 | 1.163 | −0.80 | 0.40 | 0.000 | 0.000 | 3 | 3 | 13.3 |
| forex | 8 | 1.157 | 1.211 | 0.96 | 1.38 | 0.098 | 0.023 | 0 | 0 | 29.2 |

**The verdict.**

1. **Only SPY / QQQ / IWM clear DSR > 0.95 — and they clear for *both* the OU rule and
   the control.** That is not OU alpha: a long-biased z-score MR entry on broad equity
   indices harvests the secular upward drift (their OOS spans a long bull leg), and the
   *control* in fact out-Sharpes the OU rule on all three (e.g. QQQ ctrl SR 4.81 vs OU
   2.32). No single-name sector ETF, no crypto perp, and no FX pair clears the deflation.
2. **Crypto is a loss after 7 bp/side.** Median OU SR is −1.05 annualised, median PF 0.78;
   the OU rule beats the control on DSR in only 18/27 names but both medians are ≈ 0.
   Mean reversion in a vol-scaled z-score on crypto perps does not survive realistic cost.
3. **Forex is the one place the OU rule's *raw* PF edges ahead** (median OU PF 1.157, six
   of eight pairs SR-positive), but it still fails the deflation: the best, GBPUSD, posts
   OU DSR 0.571. Cross-sectionally the OU rule does **not** dominate the control — over
   all 42 instruments the OU rule beats the control on OOS DSR only **54.8%** of the time
   and on annualised Sharpe only **42.9%**. That is a coin-flip, i.e. **no edge from the
   OU machinery over plainly tuning the thresholds in-sample.**

→ **`figures/fig3_dsr_by_market.png`** (DSR OU vs control by market),
**`fig2_oos_equity.png`** (representative OOS curves), **`fig1_ou_rule_vs_control.png`**.

---

## 6. Deepening — why the OU optimum is degenerate, and a corrected formulation

**The sl\*=3.0 degeneracy.** In the headline run the OU mesh enters LONG the spread at
**x₀ = E₀ (the long-run mean)** and the argmax cell pins **sl\* = 3.0 (the grid maximum)
for all 42 instruments**. That is not an edge: a mean-reverting process *started at its own
mean* has ≈ zero drift and a symmetric stationary band, so any first-touch rule trivially
prefers "never stop, take a small profit". But the live entry signal does **not** fire at
the mean — it fires at a z-**extreme** and bets on reversion *back* to the mean. The
headline OU mesh therefore simulates the wrong starting state.

**The fix (a third arm).** `deepen.py` adds a geometrically-correct mesh
(`ou_mesh_dev`, Numba, bit-identical-verified): start each path at the **deviation**
x₀ = E₀ ± entry_z·σ/√(1−φ²) (entry_z in stationary-std units → OU-level units), measure
profit toward the mean, profit-take at +pt of reversion, stop-loss at −sl of further
divergence. We then run **three arms** OOS with identical costs and intrabar exits:
OU-enter-at-mean / OU-enter-at-deviation / IS-tuned control.

**What the correction changes — and what it does not:**

| market | n | med DSR OU-mean | med DSR OU-dev | med DSR ctrl | OU-dev beats ctrl (DSR) | frac sl\*=3.0 (dev) |
|---|--:|--:|--:|--:|--:|--:|
| crypto | 27 | 0.0001 | 0.0003 | 0.0003 | 44% | 100% |
| equities | 7 | 0.000 | 0.0006 | 0.0034 | 43% | 86% |
| forex | 8 | 0.098 | 0.113 | 0.082 | 63% | 100% |

- The correction is **right and it does move the needle**: OU-dev beats OU-mean on OOS DSR
  in **73.8%** of instruments and on Sharpe in **81%**, and it sharply rescues the best FX
  pair — **GBPUSD OU-dev DSR = 0.945** (vs 0.571 enter-at-mean, vs 0.667 control), a near
  miss of the deflation bar. So entering at the deviation is the formulation a careful
  reader should use.
- **But the verdict is unchanged.** Still **only SPY/QQQ/IWM** clear DSR > 0.95 (all three
  arms), and OU-dev beats the IS-tuned control on DSR only **47.6%** of the time — again a
  coin-flip. The OU apparatus does not earn its complexity on this entry.
- **The degeneracy is *intrinsic*, not an entry-point artifact.** Even enter-at-deviation
  selects **sl\* = 3.0 in 41/42 instruments** (only XLV at 2.75). A first-touch rule on a
  mean-reverting process at the half-lives we measure (10–58 bars) almost always wants the
  widest admissible stop — the optimal-rule mesh has little to say beyond "don't stop
  early", which a 12×12 grid can only express as "go to the edge". This is the
  load-bearing methodological finding.

→ **`figures/fig5_three_arm_dsr.png`** (three-arm DSR), **`fig6_sl_star_degeneracy.png`**
(both arms pin the grid edge).

---

## 7. Triple Penance — the clean, citable result

This is where the chapter's machinery genuinely pays off. We measure the AR(1) φ of each
strategy's realised OOS returns, form k(φ) = (1+φ)/(1−φ), and compare three things: the
**naive IID** closed-form MaxDD, the **AR(1)-adjusted** closed-form MaxDD, and the
**realised** OOS MaxDD. (23 of 42 instruments have positive OOS drift, so a bound exists;
the rest are flagged unbounded and excluded from the bound comparison.)

- Strategy returns are **strongly positively autocorrelated** — median AR(1) φ rises from
  0.24 (crypto) to 0.37 (equities) to 0.40 (forex), and the selected arms reach φ up to
  0.74. Median variance-inflation **k = 3.28×** on the bounded set.
- **The naive IID bound understates the realised drawdown by ≈ 3.2×** (median realised
  MaxDD / IID-bound = 3.234). This is the textbook failure the Triple Penance paper warns
  about: ignore serial correlation and your drawdown budget is off by a factor of three.
- **The AR(1) correction closes the gap.** Median realised MaxDD / AR(1)-bound = 0.722 —
  the serial-correlation-adjusted bound is the right order of magnitude and, as a
  95%-confidence envelope, sits *above* the realised drawdown for most names. The
  inflation factor k(φ) almost exactly accounts for the IID bound's shortfall.

→ **`figures/fig7_tp_empirical.png`** (realised vs AR(1)-adjusted MaxDD; and the
k = (1+φ)/(1−φ) inflation curve across the panel), **`fig4_triple_penance.png`**
(IID vs AR(1) closed-form bounds).

---

## 8. Honest read & paper-worthiness

**Does the OU rule add value? No — not on a vanilla entry.** Across three asset classes,
42 instruments, 27 trials each, realistic costs and intrabar exits, the OU optimal-rule
apparatus is a coin-flip against simply tuning the PT/SL thresholds in-sample (54.8% on
DSR for the textbook formulation, 47.6% for the corrected one), and it never produces a
deflation-clearing winner that the control doesn't also produce. The only DSR > 0.95
instruments are the three broad equity indices, where the "edge" is long-side beta drift
shared by every arm — and there the dumb control wins. We report this plainly: **on this
entry signal, the OU machinery does not earn its complexity.**

**Two findings are nonetheless paper-worthy:**

1. **The sl\*=3.0 degeneracy is a genuine, reproducible critique of the Ch. 13 recipe as
   commonly applied.** The OU first-touch mesh, on a mean-reverting series at realistic
   half-lives, drives the stop to the grid edge regardless of the entry point — the
   "optimal rule" collapses to "don't stop early", which is information-free. We trace it,
   give the geometrically-correct enter-at-deviation formulation (which helps but does not
   remove it), and show the degeneracy is intrinsic. That is a clean negative result with
   a constructive correction.
2. **The Triple Penance empirical demonstration is strong and stands on its own.** A 42-
   instrument, three-asset-class panel showing the IID drawdown bound is ≈ 3× too
   optimistic and that the k(φ) = (1+φ)/(1−φ) inflation closes the gap is a compelling,
   honest illustration of the 2015 paper's result on real strategy returns.

**A serious paper would frame this as a methodology audit**: "the backtest-free OU rule is
fragile on plain entries and degenerate in its stop selection; its companion Triple
Penance drawdown correction, by contrast, is empirically vindicated." The OU half is a
cautionary negative result; the Triple Penance half is a positive one. Both are publishable
as honest reproductions; neither is a tradable alpha claim.

**Data needs: none.** The existing real 1m caches (27 crypto perps, 7 ETFs, 8 FX pairs)
are sufficient for both findings. If one wanted to give the OU rule its best shot, the
right next step is **not more data** but a *genuinely* mean-reverting entry — a fitted
residual spread from a cointegrated pair (crypto residual spreads, an equity stat-arb
pair) rather than a univariate z-score — where the OU process is the true DGP and the
optimal-rule mesh has something non-degenerate to optimise. That is a structural-axis
extension, not a data-collection one.

---

## 9. Reproduce

```bash
cd /home/daru/ldp_review/projects/12_optimal_trading_rules
bash run_full.sh                                       # headline 42-inst run
python3 scripts/run_optimal_trading_rules.py --verify   # kernel bit-identical check
python3 scripts/deepen.py                               # three-arm deepening + triple penance
python3 scripts/deepen.py --verify                      # enter-at-deviation kernel check
```

Outputs:
`tables/{per_instrument.csv, by_market_summary.csv, results.md, raw_results.parquet}`,
`tables/{deepen_per_instrument.csv, deepen_by_market.csv, deepen_summary.md, deepen_raw.parquet}`,
`figures/fig{1..7}_*.png`.
