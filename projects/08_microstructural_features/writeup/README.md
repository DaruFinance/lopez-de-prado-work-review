# Project 08 — Microstructural Features (Roll, Corwin-Schultz, Kyle, Amihud, Hasbrouck, VPIN)

**LdP source:** *Advances in Financial Machine Learning* (2018), Ch. 19 ("Microstructural
Features"). **Markets:** Crypto (10 perps), US Equities (7 ETFs), Forex (7 majors) —
information-driven dollar bars for crypto & equities, tick bars for forex (dogfooding Project 1).
**Status: DONE.** Full multi-market run + Phase-2 deepening (per-estimator attribution + a clean
data-tier ablation). Headline is the **Deflated Sharpe Ratio**, not raw Sharpe.

---

## 1. What LdP proposed (Ch. 19)

A "second generation" of microstructure estimators built from **bar data alone** (no full order
book), grouped by the model they descend from:

- **Sequential-trade / spread models** — *Roll* (1984) effective spread from the serial covariance
  of price changes; *Corwin-Schultz* (2012) high-low spread.
- **Strategic / price-impact models** — *Kyle's* λ (1985, impact per unit signed volume);
  *Amihud* (2002) illiquidity (|return| per dollar traded); *Hasbrouck* (2009) λ (impact per signed
  √dollar).
- **Volume-clock toxicity** — *VPIN* (Easley, López de Prado, O'Hara 2012), the
  volume-synchronized probability of informed trading, plus order-flow imbalance (OFI) and the tick
  rule.

The premise: these features summarise *who is trading and at what cost*, which a return/momentum
feature set cannot see, and so should add predictive content for short-horizon moves and volatility.
This project tests that premise — across three markets, causally, and with a hard overfitting gate.

## 2. Method & data

For each instrument we build ≈8 information-driven bars/day, then:

1. **Causal feature library** (`scripts/micro_features.py`). Every estimator is computed on a
   strictly-causal trailing window: the value at bar `t` uses only bars `≤ t`. Bar-level signing
   (tick rule, BVC) uses only the bar's own OHLC/volume. All rolling reductions have a pure-python
   reference and an `@njit` kernel, **verified bit-identical** (`--verify` → `max|Δ| = 0.000e+00`).

2. **Honest per-market availability tiers** — full VPIN / order-flow imbalance need
   *buyer-vs-seller* volume:
   - **Crypto** has it (`taker_buy_volume`) → **TRUE** signed flow; all 9 estimators.
   - **Equities** have volume + trade count but no side → classified with **Bulk-Volume
     Classification** (BVC) and labelled proxies; 9 estimators (4 via BVC).
   - **Forex** has neither volume nor side (tick count only) → **price-only** estimators + a
     tick-rule Kyle (impact per signed tick); VPIN / Hasbrouck / OFI **not computable**, 6 features.

   | estimator | crypto | equity | forex |
   |---|---|---|---|
   | Roll spread, Corwin-Schultz, Amihud, tick-sign, tick-flow | exact | exact | exact |
   | Kyle λ | true flow | BVC proxy | tick-rule |
   | Hasbrouck λ, VPIN, OFI | true flow | BVC proxy | — not computable |

   (Figure `fig1_availability_matrix.png`.)

3. **Predictive test.** A regularised `RandomForestClassifier` (200 trees, `max_depth=5`,
   `min_samples_leaf=50`) predicts (a) **next-bar direction** and (b) **next-bar volatility**
   (|next return| above the trailing-vol scale), scored OOS under **purged k-fold CV** + 1% embargo.

4. **Costed trading test.** A long/short on the direction signal, charging a **realistic per-market
   cost** on every position change — **via `lib/realism.py`, no clamping**: equity = time-of-day
   half-spread schedule (bp) + per-fill min-ticket commission; forex = UTC-hour half-spread schedule
   in pips; crypto = house default (8 bp round-trip). Net-of-cost per-fold Sharpe trials feed the
   **Deflated Sharpe Ratio + PBO (CSCV)** from `lib/overfit.py`.

**Causality:** the only forward-looking objects are the two labels; purged CV protects them.

## 3. Engineering

- **Profile (1 core).** `RandomForest.fit` is ~96% of wall time; the only non-sklearn hot loop is
  the rolling microstructure estimators (Roll / Kyle / Hasbrouck / Amihud / VPIN).
- **Numba.** Those rolling reductions are `@njit` kernels, verified bit-identical to the pure-python
  references (`max|Δ| = 0.000e+00`); rolling block ≈ **475×** faster on real bars. Corwin-Schultz is
  a fixed 2-bar formula → kept as vectorised numpy (avoids a 1-ULP libm-vs-LLVM `log`/`exp`
  mismatch). sklearn is not Numba'd, per house rule.
- **Phase-2 parallelism.** The deepening's single-feature sweep is many tiny fits, so it parallelises
  *across instruments* (loky pool, one thread per worker) rather than threading each fit — a
  process-pool over 24 instruments runs the full deepening in ~3.5 min. Results are deterministic
  (`random_state=0`), bit-stable across the n_jobs=1 smoke and the parallel full run.

## 4. Headline results

### 4a. Predictive content — volatility yes (weakly), direction no
(`tables/micro_by_market.md`, `fig2_predictive_auc_by_market.png`)

| market | instruments | n_obs (median) | dir AUC | vol AUC | net Sharpe | net mean (bp/bar) |
|---|---:|---:|---:|---:|---:|---:|
| crypto | 10 | 8,709 | 0.510 | 0.538 | −0.018 | −2.72 |
| equity | 7  | 42,397 | 0.513 | 0.551 | −0.355 | −33.45 |
| forex  | 7  | 8,694 | 0.507 | 0.539 | −0.016 | −0.26 |

- **Next-bar volatility AUC is consistently > 0.50** (≈0.538–0.551, all three markets, tight error
  bars) — microstructure features carry real short-horizon *volatility* information. This is the one
  positive, robust finding.
- **Next-bar direction AUC sits at the coin flip** (≈0.507–0.513, error bars touching 0.50).
  Microstructure features do **not** time direction at the bar horizon in any market.

### 4b. Which estimator carries the signal (DEEPENED)
Single-feature purged-CV AUC, one estimator at a time (`tables/micro_per_estimator.csv`,
`fig5_per_estimator_auc.png`):

| | top-2 estimators for **next-bar VOLATILITY** (single-feature OOS AUC) |
|---|---|
| **crypto** | `amihud` 0.523, `hasbrouck` 0.520 |
| **equity** | `vpin` 0.531, `ofi` 0.523 |
| **forex**  | `kyle` 0.535, `amihud` 0.532 |

For **DIRECTION**, *every* estimator sits at 0.498–0.512 in *every* market — there is no
single-feature direction edge anywhere. The vol signal is carried by **different estimators in
different markets**: illiquidity/impact (`amihud`, `hasbrouck`, `kyle`) in crypto/forex, but the
flow-toxicity estimators (`vpin`, `ofi`) in equities. That the equity flow estimators are computed
from a *proxy* (BVC) and still top the table is the bridge to the next result.

### 4c. How much does the data tier matter? (DEEPENED — the cleanest finding)
The Phase-1 cross-market comparison confounds the data tier with the instrument. The clean
experiment runs on **identical crypto bars** — the only market with true side volume — recomputing
the four side-volume estimators three ways and measuring the vol-AUC of that 4-feature block
(`tables/micro_tier_ablation.csv`, `tables/micro_tier_gap.csv`, `fig6_tier_ablation.png`):

| tier | what it represents | mean vol AUC (10 crypto instr) |
|---|---|---:|
| **TRUE** side-volume | what crypto actually has | 0.522 |
| **BVC** proxy | the *equity* tier (price+volume, no side) | **0.547** |
| **tick-only** | the *forex* tier (price only) | 0.526 |

**The expensive true-flow tier buys nothing for next-bar volatility.** On 9 of 10 crypto
instruments the free BVC proxy *matches or beats* the true buy/sell split (mean gap **−0.024 AUC in
BVC's favour**), and even the tick-only signing is competitive (within 0.004 of true flow). The
predictive content lives in the **price/volume dynamics that BVC reconstructs**, not in the actual
trade-side data. For *direction*, all three tiers sit at the coin flip (0.507/0.512/0.507).

This is the project's most useful practical conclusion: **a research desk does not need true
order-flow data to extract the (modest) microstructure vol signal** — a BVC proxy off OHLCV is as
good, which is exactly why the equity tier (no side data) was not handicapped in §4a.

### 4d. Honest DSR-gated verdict — nothing tradeable after costs
(`tables/micro_dsr_pbo.csv`, `fig3_costed_sharpe_dsr.png`, `fig4_pbo_distribution.png`)

The costed long/short on the direction signal is **net-negative in 22 of 24 instruments** (only
TRXUSDT and EURUSD scrape marginally positive, both per-bar Sharpe < 0.01). Equities are the
worst (net −33 bp/bar) because the min-ticket commission dominates a 1-unit book; crypto and forex
sit just below zero.

The overfitting gate is the headline:

| scope | n trials | best per-bar SR | E[max] benchmark (SR0) | **DSR** | PBO |
|---|---:|---:|---:|---:|---:|
| all | 144 | 0.070 | 0.535 | **0.00** | 0.10 |
| crypto | 60 | 0.070 | 0.057 | 0.89 | 0.36 |
| equity | 42 | −0.032 | 0.474 | **0.00** | 0.02 |
| forex | 42 | 0.053 | 0.054 | 0.45 | 0.13 |

Read carefully: the *all-scope* DSR is **0.00** — the best of 144 trials does not clear the
expected-max benchmark implied by the trial dispersion, i.e. it is consistent with luck. The
crypto-scope DSR of 0.89 looks high only because it deflates against a tiny same-scope benchmark
(60 near-zero trials, best SR 0.07) — and its **PBO of 0.36** says the best in-sample fold is in the
bottom half out-of-sample more than a third of the time. Across the whole study, **the direction
signal is not tradeable after realistic costs**, exactly the skeptical reading Ch.19 + the DSR gate
are designed to produce.

## 5. Independent opinion / extensions beyond LdP

- **Ch.19's framing oversells direction.** On bar data alone, the microstructure estimators carry
  *volatility/activity* information, not *directional* information. The honest claim is
  "these features help size and time risk, not pick sides."
- **The data-tier ablation (§4c) is, to my knowledge, not in the book and is the most interesting
  result here.** It quantifies that BVC ≈ true flow for the vol task on identical instruments, which
  reframes the "you need order-flow data" intuition: you need *price+volume*, and BVC bridges the gap.
  This is the natural seed for a short methods note (see §8).
- **Vol-AUC, not direction-AUC, is where any future work should aim** — e.g. feeding these features
  into a volatility-targeting overlay or a meta-label (Project 03) rather than a standalone
  long/short. That is a clean follow-on but out of scope here.

## 6. Reproduce

```bash
python3 scripts/run_microstructure.py --verify    # bit-identical kernel check (max|Δ|=0)
python3 scripts/run_microstructure.py --profile   # cProfile + Numba speedup
python3 scripts/run_microstructure.py --smoke      # 1-core tiny-subset end-to-end
bash run_full.sh                                   # full multi-market Phase-1 run
python3 scripts/deepen_microstructure.py           # Phase-2: per-estimator + tier ablation
python3 scripts/deepen_microstructure.py --smoke   # fast deepening sanity
```

**Outputs** — `tables/`: `micro_availability.csv`, `micro_predictive.csv`, `micro_by_market.csv/.md`,
`micro_dsr_pbo.csv`, `micro_per_estimator.csv`, `micro_tier_ablation.csv`, `micro_tier_gap.csv`;
`figures/`: `fig1_availability_matrix`, `fig2_predictive_auc_by_market`, `fig3_costed_sharpe_dsr`,
`fig4_pbo_distribution`, `fig5_per_estimator_auc`, `fig6_tier_ablation`.

## 7. Limitations / honest caveats

- **No order book.** Every estimator is a bar-data proxy by design (Ch.19's whole point); the true
  Kyle/Hasbrouck λ from quote-level data is not tested here.
- **Forex tier is the thinnest.** Spot FX has no volume; the FX VPIN/Hasbrouck/OFI columns are
  *absent*, not zero-filled, and the FX cost schedule is a calibrated retail half-spread (the
  HistData fetch discarded per-tick spread) — documented in `lib/realism.py`.
- **Bar-horizon only.** The labels are next-*bar*; a longer label span (Project 03 triple-barrier)
  could change the direction result. Not tested.
- **Costs are realistic, not adversarial.** Equity uses a min-ticket commission on a 1-unit book,
  which is punishing for a tiny notional; a larger book would dilute the per-bar commission. The
  *qualitative* verdict (DSR ≈ 0 all-scope) is robust to this because direction AUC ≈ 0.50 leaves no
  gross edge to defend even before costs.

## 8. Paper-worthiness

**Not a standalone paper on its own headline** (the direction result is null and the vol result is
modest). **But §4c — the data-tier ablation — is a genuinely publishable methods note:** *"For
bar-level microstructure features, Bulk-Volume Classification recovers the next-bar volatility signal
as well as true trade-side data; the order-flow tier is not the binding constraint."* It is clean
(same instruments, three tiers, one controlled variable), causal, multi-instrument (10 crypto),
DSR-/AUC-gated, and counter to a common practitioner assumption. The strongest packaging is a short
note built around fig5 + fig6, with the DSR gate (§4d) as the sobriety check that keeps it honest —
i.e. "here is exactly how much (little) the expensive data tier is worth, and here is why none of it
is tradeable as a standalone signal." That is the form this should take if it graduates.
