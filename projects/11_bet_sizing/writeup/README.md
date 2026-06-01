# Bet Sizing from Predicted Probabilities

**López de Prado, *Advances in Financial Machine Learning*, Chapter 10**
Project 11 of the LdP review program · 42 instruments · crypto + US equities + forex

---

## 1. Question

Chapter 10 of *Advances in Financial Machine Learning* makes a specific operational
claim: once a secondary ("meta") classifier outputs a calibrated probability that the
primary strategy's bet will be profitable, you should **size the bet by that
probability** rather than trading a fixed unit. López de Prado gives an explicit recipe
, map the probability to a signed size through a normal-CDF transform, average the
sizes of all *concurrently active* bets into a single net book position, and optionally
**discretize** that size onto a coarse grid to stop the book from churning on tiny
size changes.

The recipe is presented as a way to translate a classifier's confidence into a
position. The open question, the one the chapter does not settle empirically, is
whether this layer **adds risk-adjusted, overfitting-deflated performance**, or whether
it is purely a **precision / execution layer** that changes *how* a fixed edge is
expressed (mainly by cutting turnover) without changing *whether* there is an edge.

This project tests that distinction at scale. We hold the primary strategy, the
triple-barrier labels, the meta-model, the costs, and the active-bet-averaging engine
**identical** across three sizing schemes and vary only the per-event target size:

- **fixed**, signed unit size (`s · 1`) on every acted event;
- **prob**, continuous size `s · m`, `m = 2Φ(z) − 1`, `z = (p − ½)/√(p(1−p))`;
- **prob_disc**, the same `m` snapped to a grid step `d` and clipped to `[−1, 1]`.

The **headline metric is the Deflated Sharpe Ratio (DSR)**, net of realistic costs,
because the whole point of the experiment is to ask whether any *apparent* sizing
benefit survives the multiple-testing deflation implied by tuning the knobs. Raw PF and
Sharpe are reported as diagnostics only; turnover is reported as the mechanism.

---

## 2. Data

All results are on **real 1-minute data**, no synthetic series, aggregated to ~20,000
information-driven bars per instrument (dollar bars for crypto/equity, count-threshold
bars for FX, matching the rest of the program).

| Market | Instruments | Source | Span |
|---|---|---|---|
| Crypto | 27 USDT perps (BTC, ETH, SOL, … SHIB) | Binance 1m | full available history |
| US equities | 7 ETFs (SPY, QQQ, IWM, XLK, XLF, XLE, XLV) | AlgoSeek 1m, RTH only | 2007-2026 |
| Forex | 8 majors (EURUSD, USDJPY, GBPUSD, … EURGBP) | HistData 1m | full available history |

Per instrument the median event count is **291** triple-barrier events from the EMA
crossover primary, small enough that the deflation question has real teeth, large
enough to fit a bagged-tree meta-model out-of-fold.

**Costs are realistic and never clamped.** Crypto carries the program house cost (7 bp
per side). Equity and forex frictions come from `lib/realism.py`: a causal, ex-ante
**time-of-day half-spread schedule** (FX: London-NY overlap tightest, rollover/Asian
wide; equity: open/close wide, midday tight) plus broker commission. Weekend and
overnight **gaps are kept as real risk**, not floored.

> **One correction made in this phase (documented for the audit trail).**
> `lib/realism.equity_commission_rate_bp` models a literal **1-share** position, so its
> $0.35 min-ticket floor becomes **8.75 bp/fill on a $400 SPY share** (70 bp on a $50
> share), an artifact of a degenerate 1-share book, not a real friction. In the
> initial run this crushed every equity trade's net label, so the meta-model learned
> `p ≈ 0.2`, almost nothing crossed the act-gate (`frac_act ≈ 0`), and 5 of 7 ETFs
> produced NaN/zero books. The fix (project-local, lib left read-only) applies the
> **same** min-ticket schedule and the **same** `lib/realism` spread schedule on a
> realistic **$10,000** position: commission falls to ≈0.35 bp/fill and equities become
> informative (cost 0.85-3 bp/side, `frac_act` 1-56%). No clamping, costs still fall
> where they fall; we only sized the commission on a non-degenerate book. See
> `scripts/run_bet_sizing.py:equity_per_side_cost_realistic`.

---

## 3. Method

**Primary → labels.** An EMA fast/slow crossover fixes the **side** `s ∈ {+1, −1}`. Each
crossover opens a triple-barrier trade (volatility-scaled profit-take / stop / max-hold,
all evaluated on full OHLC). The meta-label is `y = 1{net-of-cost trade PnL > 0}`.

**Meta → probability (leakage-free).** A bagged decision-tree classifier
(`BaggingClassifier(DecisionTree(max_depth=4), 40 estimators`) produces `p = P(y = 1)`
**out-of-fold** via **purged k-fold CV with embargo** (6 folds, 1% embargo, label-span
purge), reusing the program's `projects/03_meta_labeling` triple-barrier library so
overlapping labels cannot leak into the sizing input. This sklearn fit dominates runtime
(~75% of wall time) and is deliberately left un-Numba'd.

**Size → book.** `p` maps to a magnitude `m` (eq. 10.1-10.2); the act-gate vetoes events
with `p < meta_threshold`. The three schemes set the per-event target size, then the
**identical** active-bet-averaging kernel turns overlapping `[entry, entry+hold)`
windows into a single per-bar net book position (`avgActiveSignals`). That kernel is a
Numba difference-array implementation, **verified bit-identical** (`max|Δ| = 0` over 200
random cases) against an independent pure-Python reference and ~102× faster at scale.

**Costed P&L.** `pnl_t = pos_t · r_{t→t+1} − cost_t · |pos_t − pos_{t−1}|`, cost charged
on the *change* in book position (= realised turnover), causal throughout.

**Knobs as trials.** The structural shape is fixed; the numeric knobs
`(fast/slow, pt/sl, max_hold, meta_threshold, disc_step)` form a **32-point IS-tunable
grid**. These 32 are the trials fed to the **False Strategy Theorem** for the DSR
benchmark, to **PBO** (CSCV) on the prob-sized trial matrix, and to the **effective-N**
(eigenvalue participation ratio) deflation diagnostics.

---

## 4. Results

### 4.1 The robust effect, turnover collapses everywhere

Probability sizing **drastically cuts book turnover** in every market and for nearly
every instrument. All-instrument median book turnover falls **254 → 48.6 (prob, −81%)
→ 46.2 (disc, −82%)**:

| Market | n | med turnover fixed | prob | disc | prob cut | disc cut |
|---|---:|---:|---:|---:|---:|---:|
| crypto | 27 | 272 | 53.6 | 52.3 | −80% | −84% |
| equities | 7 | 197 | 20.4 | 19.6 | −87% | −87% |
| forex | 8 | 173 | 42.2 | 29.0 | −84% | −86% |

All 42 instruments sit below the fixed-size baseline (`fig5_turnover_collapse.png`);
the typical turnover ratio is 0.13-0.25× fixed. The mechanism is exactly LdP's: averaging
overlapping bets nets opposing positions and damps the constant ±1 flipping of a
fixed-unit book, and discretization further suppresses the long tail of tiny
near-zero-conviction size changes that otherwise pay cost without expressing a view.

### 4.2 The headline, deflated performance does **not** move

The DSR is statistically indistinguishable between schemes. Median per-market DSR:

| Market | fixed | prob | disc |
|---|---:|---:|---:|
| crypto | 0.027 | 0.013 | 0.032 |
| equities | 0.321 | 0.254 | 0.313 |
| forex | 0.087 | 0.113 | 0.186 |

Paired per-instrument deltas (Δ = scheme − fixed), with a two-sided sign test:

| Δ vs fixed | median ΔDSR | frac instruments > 0 | sign-test p |
|---|---:|---:|---:|
| prob − fixed (all 42) | **−0.0022** | 0.40 | 0.28 |
| disc − fixed (all 42) | **+0.0026** | 0.55 | 0.64 |

Both deltas are near zero and **not significant**, no market reaches p < 0.45. And the
decisive number: **0 of 42 instruments cross DSR > 0.95 under *any* scheme** (fixed,
prob, or disc alike). The best instruments (USDJPY 0.65, EURUSD 0.56, XLE 0.58 fixed)
are well short of significance, and sizing neither lifts them over the bar nor pushes
the field there. **PBO median 0.49** (a coin-flip) confirms the trial-selection has no
IS→OOS persistence; **effective-N is 2-4** (median 3) of the 32 knob trials, so the
deflation benchmark is appropriately strict.

### 4.3 The one real ordering, discretized ≥ continuous

Where the two sizing variants differ, **discretization is the better one**: `disc_dsr ≥
prob_dsr` in **33 of 42** instruments, and discretized PF beats fixed in **26/42** vs
only **15/42** for continuous prob. Continuous prob sizing's median ΔDSR is slightly
*negative*; discretized is slightly *positive* (`fig6_dsr_delta.png`). Interpretation:
continuous `m` keeps a cloud of tiny bets (`fig4_betsize_dist.png`) that each pay
turnover cost without conviction; snapping to a 0.1 grid zeros the smallest of them,
which is a mild net positive. This is a **cost-efficiency** ordering, not an alpha one,
the gap is a fraction of a DSR point.

---

## 5. Interpretation, precision layer, not alpha

The evidence is consistent and one-directional:

1. **Turnover collapse is large, universal, and significant** (−80 to −87%, all 42
   instruments). This is a genuine, repeatable property of probability sizing + active
   averaging.
2. **Deflated performance does not change** (ΔDSR ≈ 0, no sign-test significance, 0/42
   over the 0.95 bar). Sizing redistributes *how* the book expresses its position; it
   does not create edge.
3. The only consistent ranking is **discretized ≳ continuous**, and that too is a
   cost-efficiency effect (fewer pointless micro-trades), not an alpha effect.

The honest reading: **bet sizing from probabilities is a precision / cost-control layer,
not an alpha source.** If the primary strategy has a real, deflatable edge, sizing lets
you express it more cheaply and with less churn, which in a *cost-dominated* regime
(crypto at 7 bp, FX rollover, equity open/close spreads) is worth having. But it cannot
manufacture significance where the primary has none. Here the EMA-crossover primary has
no deflatable edge to amplify (every scheme's DSR sits near zero), so sizing's only
visible footprint is the turnover line. That is precisely the prediction of treating
Ch.10 as an execution layer rather than a signal.

This does **not** contradict López de Prado, Ch.10 is explicitly about *converting* a
classifier's confidence into a position size, not about *generating* the confidence. The
result simply makes the boundary empirical: the value of the chapter is realised at the
**cost/turnover** margin, and is contingent on the meta-model upstream actually carrying
information.

---

## 6. Honest DSR verdict

- **No instrument, in any market, under any sizing scheme, has a Deflated Sharpe Ratio
  above 0.95.** After accounting for the 32 tuned trials (effective-N ≈ 3) and realistic
  costs, none of these books is statistically distinguishable from the best of 32
  skill-less trials.
- The strongest survivors are forex (USDJPY DSR ≈ 0.65-0.78 across schemes, EURUSD ≈
  0.53-0.59, GBPUSD ≈ 0.47-0.50) and a few equities (XLE, XLF, IWM ≈ 0.3), suggestive,
  not significant.
- **PBO ≈ 0.49** (median) means the in-sample-best trial lands in the OOS bottom half
  about half the time: trial selection here is essentially a coin flip, the textbook
  signature of overfitting the knob grid, and the reason the raw Sharpe/PF numbers must
  not be read as the headline.
- Therefore the project's headline is a **null result on the alpha question and a clean
  positive on the turnover question**, exactly the kind of separation the DSR
  discipline is designed to expose.

**Caveats kept in the open:** QQQ's eye-catching `prob_pf = 12.7` is an artifact of a
near-empty book, only ~4 events cleared its act-gate (`frac_act = 1.2%`), so its PF is
statistically meaningless and is excluded from any qualitative claim (its DSR, 0.27, is
unremarkable). SPY/QQQ have low base rates on this primary even at realistic cost; that
is a property of large-cap index ETFs under an EMA-crossover, not a bug.

---

## 7. Paper-worthiness

**As a standalone paper: no.** The alpha result is a null and the primary is a toy
crossover, so there is no novel edge to report.

**As one chapter of the LdP-review monograph: yes, and it earns its place.** Its
contribution is methodological rather than a strategy:

- A **clean, large-scale separation** of the two things Ch.10 actually does, confirming
  the turnover/cost benefit (−80 to −87%, 42/42 instruments, three asset classes) while
  showing the deflated-performance benefit is **zero** under honest accounting. That
  separation is rarely stated this sharply in print.
- A **reusable, verified engine**: the active-bet-averaging kernel is bit-identical to an
  independent reference and ~102× faster, and the whole pipeline shares events, OOF
  probabilities, costs, and the costed-book engine across schemes so the comparison is
  apples-to-apples.
- A **multi-market, realistic-cost, DSR-headline** treatment with PBO and effective-N
  deflation, directly demonstrating *why* raw PF (QQQ 12.7!) is dangerous and the DSR
  is the right lens.
- A documented **realism correction** (the 1-share min-ticket artifact) that is itself a
  cautionary tale about cost modeling.

The path to a *standalone* contribution would be to feed this sizing layer a primary
with a **genuine deflatable edge** (e.g. one of the program's surviving cross-pair
strategies) and measure whether the −80% turnover translates into a DSR lift once there
is real edge to express. That is the natural follow-up; it is out of scope here.

---

## 8. Reproduce / artifacts

```bash
cd projects/11_bet_sizing
./run_full.sh                                  # full 42-instrument run (~11 min, 1 core)
python3 scripts/run_bet_sizing.py --verify-kernel   # numba vs python: max|Δ| = 0
python3 scripts/deepen.py                       # paired deltas, sign tests, figs 5-6
```

**Tables** (`tables/`)
- `results.md` / `by_market_summary.csv`, by-market medians, DSR>0.95 counts, PBO, turnover
- `per_instrument.csv` / `raw_results.parquet`, per-instrument PF/SR/DSR/turnover ×3 schemes
- `deepen_paired.csv`, per-instrument paired Δ (prob−fixed, disc−fixed) + turnover ratios
- `deepen_by_market.md` / `deepen_by_market.csv`, by-market paired medians + sign-test p
- `deepen_summary.txt`, the plain-text honest read

**Figures** (`figures/`)
- `fig1_dsr_by_scheme.png`, DSR by market ×3 schemes (none reach 0.95)
- `fig2_turnover_by_scheme.png`, turnover by scheme (the collapse)
- `fig3_equity_by_scheme.png`, representative book equity curves
- `fig4_betsize_dist.png`, continuous vs discretized bet-size histograms
- `fig5_turnover_collapse.png`, per-instrument turnover ratio vs fixed (all 42 < 1)
- `fig6_dsr_delta.png`, median ΔDSR vs fixed, by market (≈ 0)

**Runtime / RAM.** Serial, single-core, `n_jobs=1`, ~673 s wall, peak ~0.85 GB resident
(one instrument in memory at a time; no N×N materialisation).
