# Cross-Sectional Machine Learning: the program's first deflation-surviving ML edge

**Sources.** *Advances in Financial Machine Learning* (2018, Ch. 7 purged
cross-validation, Ch. 8 feature importance); *The Deflated Sharpe Ratio* (2014);
*The Probability of Backtest Overfitting* (2017); *A Data Science Solution to the
Multiple-Testing Crisis* (2019). The recurring point in this literature: machine
learning adds value when it predicts the **cross-section** of returns under proper
validation, and far less when it is asked to forecast a single series' direction.

**Markets.** Crypto (a large survivorship-honest perpetual-swap universe of 787
pairs); a bonus traditional-markets check on FX majors and US-equity sector ETFs.

**Role.** Closes the program's only structural gap, which had no cross-sectional
coverage, and delivers the program's first deflation-surviving ML edge.

---

## 0. The claim under test, and the headline

Every ML result earlier in this program lived in machine learning's *weakest*
regime: single-series direction forecasting. It failed deflation, exactly as the
literature predicts. The fair test of the program's "nothing clears deflated
significance" through-line is to run the **identical** apparatus in ML's *strongest*
regime, cross-sectional return prediction, where each bar you rank the whole
universe and rotate capital into the best names market-neutral, and see whether the
verdict flips. It does, partially. That is the result worth publishing.

The same purged-CV, Deflated-Sharpe-Ratio, Probability-of-Backtest-Overfitting, and
effective-N apparatus was run on both regimes:

| regime | family | PBO | DSR survival | IS to OOS rank rho | median OOS PF | verdict |
|---|---|---:|---|---:|---:|---|
| **single-series** (ML weakest) | pooled band corpus, 25,162 strategies | **0.62** | **0 of 25,162** | n/a | ~1.0 | fails deflation |
| **cross-sectional** (ML strongest) | gradient-boosted trees, 10 horizons | **0.10** | **10 of 10 horizons** (lgbm) | **+0.78** | **1.123** | **survives deflation** |

The deflation result is corroborated across three independent tree families on the
same panel and the same walk-forward (Section 2). It also *nuances* the headline
honestly rather than overturning it: cross-sectional ML **approaches but does not
beat** the best static structural carry/momentum archetype (about PF **1.17**). The
publishable claim is therefore stronger and more defensible than "nothing works":

> **ML edge is approximately equal to the static edge; neither dominates.** Machine
> learning earns a real, deflation-surviving cross-sectional edge, but it lands at
> roughly the same place a well-built static carry/momentum strategy already sits.
> The value of ML here is *parity through a different, less crowded route*, not a
> free lunch over structure.

Figures: `figures/fig1_regime_dsr_pbo.png` (PBO and DSR survival side by side);
`figures/fig2_xs_per_horizon.png` (the cross-sectional edge by horizon, lgbm);
`figures/fig3_family_dsr.png` (OOS profit factor and OOS Sharpe by tree family and
horizon, against the deflation bar); `figures/fig4_regime_dsr_survival.png` (the
regime survival contrast).

---

## 1. What the literature says, and what we found

**What the literature says.**
- ML's defensible use is predicting the **cross-section** of returns with rigorous
  validation; single-series price or direction forecasting is the weak game.
- Any backtested edge must clear the **Deflated Sharpe Ratio** against the number
  and dispersion of trials, and the **Probability of Backtest Overfitting** must be
  low for the in-sample ranking to mean anything out of sample.
- Trees beat deep nets on tabular financial data far more often than the reverse.

**What we found.**
- The regime split is exactly as described. Single-series ML is a coin-flip or worse
  out of sample (PBO 0.62) and **nothing** survives deflation. Cross-sectional ML
  has a low PBO (0.10), strong in-sample to out-of-sample rank persistence
  (rho +0.78), and survives the Deflated Sharpe bar on every horizon tested for the
  reference model.
- The cross-sectional edge is real but **modest and bounded**: median OOS profit
  factor 1.123, annualized OOS Sharpe in the low single digits per horizon under the
  engine's own annualization, net of the full per-fill cost model. It reaches but
  does not exceed the best static structural archetype (about PF 1.17).
- Trees won, and now there is a same-task head-to-head to show it. The deflation-
  surviving families are gradient-boosted trees. Two neural families (a per-bar feed-
  forward network and a per-pair recurrent network) were run on the identical panel,
  label, walk-forward, sizing search, and cost model; both lose money net of costs
  (profit factor at or below 1, negative Sharpe) and neither clears the deflation bar
  the trees clear. The "trees beat nets on tabular cross-sections" finding is therefore
  confirmed on our own data, not just cited (Section 8). The remaining neural families
  are deferred (Section 5).

---

## 2. The deflation verdict, family by family

The cross-sectional sweep is treated as one multiple-testing family. Under the False
Strategy Theorem, the expected **maximum** Sharpe a skill-less searcher would post
across N trials is computed from the dispersion of the realized per-bar OOS Sharpe
estimates. A (family, horizon) combo "clears" if its OOS Sharpe exceeds that bar.
We report the bar at three defensible trial counts and headline N=40, a round count
for the realized tree sweep of 30 (family, horizon) combos plus the linear and
forest configurations launched in the same search.

| trial count N | expected-max skill-less Sharpe (annualized) | tree combos clearing |
|---:|---:|---:|
| 30 | ~1.67 | 23 of 30 |
| 40 | ~1.76 | 23 of 30 |
| 80 | ~1.97 | 20 of 30 |

At the headline N=40:

| family | horizons clearing the deflation bar |
|---|---|
| lgbm | **10 of 10** |
| catboost | **9 of 10** (only the longest horizon falls below) |
| xgb | **5 of 10** |

That is **23 of 30** tree (family, horizon) combos clearing a genuine multiple-
testing bar, versus **0 of about 25,162** single-series strategies. The full row-by-
row table, with per-bar and annualized Sharpe and the profit factor of every combo,
is in `tables/family_dsr.md`.

For the reference model (lgbm) we additionally report the **rigorous** per-horizon
Deflated Sharpe Ratio computed from its clean per-bar ledger: the probabilistic
Sharpe against the expected-max benchmark, accounting for the heavy skew and
kurtosis of the rotation returns, deflated against the dispersion of the 10-horizon
Sharpe family. All ten horizons clear DSR > 0.95 (DSR rounds to 1.00 on nine of ten
and 0.9999 on the tenth). The headline PBO (0.10) and rank persistence (+0.78) are
the canonical per-window figures from the dedicated rotation engine, which is the
correct trial axis. A PBO recomputed across only the ten horizon columns of one
model is near-degenerate by construction (it reads 0.62 precisely because the
horizons are near-identical bets of the same model); that cross-check is reported,
clearly labelled, in `tables/family_dsr.md` and is *not* the headline.

---

## 3. The cross-sectional edge, horizon by horizon

The cross-sectional strategy is one rotating long-short portfolio per forecast
horizon: each bar, rank the alive universe by a causal per-name tree score, go long
the top quantile and short the bottom, market-neutral, with the bet-sizing knobs
(quantile, long-only, tilt, rebalance throttle) tuned in-sample per walk-forward
window. Ten horizons were searched; they are part of the trial family the deflation
bar accounts for.

| H (bars) | OOS PF (lgbm) | OOS Sharpe (annualized) | DSR | survives DSR > 0.95 |
|---:|---:|---:|---:|:--:|
| 4 | 1.148 | 3.23 | ~1.00 | yes |
| 6 | 1.127 | 2.79 | ~1.00 | yes |
| 8 | 1.120 | 2.61 | ~1.00 | yes |
| 12 | 1.158 | 2.66 | ~1.00 | yes |
| 24 | 1.149 | 3.62 | ~1.00 | yes |
| 48 | 1.111 | 2.62 | ~1.00 | yes |
| 72 | 1.140 | 3.36 | ~1.00 | yes |
| 120 | 1.107 | 2.56 | ~1.00 | yes |
| 168 | 1.095 | 2.39 | ~1.00 | yes |
| 336 | 1.089 | 2.19 | ~1.00 | yes |

Full numbers: `tables/xs_per_horizon.md` and `tables/family_dsr.md`. The edge is
strongest at short to medium horizons and decays gently as H grows, consistent with
a momentum and reversal cross-section rather than a single lucky horizon. The
annualized Sharpe figures use the engine's own bar-frequency annualization and are
indicative; rotation returns are autocorrelated within a holding period, so the
effective number of independent observations, and the honest Sharpe, are lower than
the raw bar count implies. The deflation verdict and the profit factor do not depend
on that annualization choice.

---

## 4. Why single-series fails and cross-sectional survives

Same models, same triple-barrier labeling, same purged walk-forward, same cost
model. The difference is **the prediction target**, and it is the difference the
literature predicts:

- **Single-series direction** asks one noisy series whether it goes up or down. The
  signal-to-noise is brutal; the program's audit put the *effective* number of
  independent bets at about 40 out of 25,162 nominal strategies, and the in-sample
  best lands in the out-of-sample bottom half 62% of the time. Deflation kills all of
  it.
- **Cross-sectional** asks a much easier, more stable question, which names will
  out-perform which, and aggregates across a wide universe each bar, which averages
  down idiosyncratic noise and yields a ranking that persists out of sample
  (rho +0.78). That persistence is exactly what lets it clear the deflation bar.

---

## 5. Honest limits and what still needs a heavy run

- **Three tree families completed; some neural families now have real ledgers, the
  rest are deferred.** lgbm, xgb, and catboost ran end-to-end on the full 787-pair
  panel with verified out-of-sample artefacts. Random forest and extra-trees were cut
  as impractically slow and memory-heavy on the full panel. The linear families
  (elastic-net, ridge, lasso) were stopped by the memory guard partway through the
  search. The neural families need a larger accelerator than the local card (the full
  panel exceeds local GPU memory), so they were run on a managed 96 GB accelerator.
  The per-bar feed-forward network (mlp) and the recurrent network (lstm) now have
  real, costed, purged-walk-forward out-of-sample ledgers on the identical panel,
  label, and cost model as the trees; their numbers are reported in Section 8 and in
  `tables/family_dsr.md`. The remaining neural families (the gated-recurrent and
  temporal-convolution sets, and the cross-pair attention set) did not produce a
  usable ledger on this pass and are recorded as deferred; none of the deferred
  families carries a number here.
- **The edge is bounded.** A median OOS profit factor of 1.123 net of costs is real,
  but it is parity with static structure, not a dominating alpha. The honest framing
  is that ML reaches the static frontier by a different route, and that is the claim
  this writeup makes.
- **Cross-checks are labelled.** Where a statistic is recomputed on a different trial
  axis than the dedicated engine, the canonical per-window figure is the headline and
  the recompute is labelled a cross-check; they are not conflated.

---

## 6. Bonus: the pattern is not purely a crypto artefact

A separate, self-contained cross-sectional engine runs the same idea on traditional
markets: eight FX majors and nine US-equity sector and index ETFs (daily currency
and sector rotation, realistically costed, purged walk-forward, deflation-gated).
The cross-section there is *thin* (a quarter-quantile is about two names a side), so
the breadth is modest and complementary rather than a volume driver, but it lets the
rotation form be tested outside crypto. Smoke runs for both FX and equity are banked
(`tables/multimarket_xsection.md`); on these tiny runs (two families, two horizons, a
handful of names a side) neither clears deflation, exactly as expected at that scale.
A full FX and equity run is the cheap follow-up that would confirm the regime story
is asset-class-general rather than crypto-specific.

---

## 7. Multi-market: does the cross-sectional ML edge hold in equities too?

The crypto cross-section is a wide panel of hundreds of liquid instruments. To check
whether the cross-sectional ML edge is a property of that one market or a property of
the rotation idea, we ran the identical recipe on the canonical equity cross-section:
a bounded, liquid universe of large-cap US stocks (members of the broad large-cap
index), adjusted daily bars over roughly eleven years. Each bar we rank the live
universe by a causal per-name tree score, rotate long the top quantile and short the
bottom quantile (market-neutral), tune the rotation knobs in-sample inside each
purged walk-forward window, and charge realistic per-fill costs. The horizon family
is the standard set of one day, one week, two weeks, one month, and one quarter. The
scoring apparatus is the same deflation toolkit used everywhere else in this program.

The universe is about two hundred and forty names with a median of roughly two
hundred and forty-five live each bar, so this is a genuinely wide cross-section (a
quarter-quantile is dozens of names a side, unlike the thin currency and sector
rotations of the previous section). The model family is the gradient-boosted tree
that won in crypto.

**Result.** The raw edge survives the move to equities, but it is thinner. The
profit factor is in the same neighbourhood as crypto: a median out-of-sample profit
factor of about 1.11 across the five horizons (crypto: about 1.12), peaking near 1.15
at the two-week horizon. Direction and ordering are sensible: the medium horizons
(two weeks to one month) are the strongest and the very short and very long horizons
are weakest. But the annualised Sharpe ratios are far lower than crypto (roughly 0.3
to 0.7 versus crypto's roughly 2.7), because a daily equity rotation has far fewer
rebalances and a much smaller effective sample than an hourly rotation over a much
larger panel. As a direct consequence the edge does **not** clear the deflation bar:
zero of five horizons reach a Deflated Sharpe above 0.95, although the best horizon
gets close (about 0.94 at two weeks, about 0.91 at one month). The very short horizon
also carries heavy tails (high kurtosis), which further penalises its deflated score.

**Reading.** The cross-sectional ML edge **partially replicates** in US equities: the
sign and magnitude of the profit factor carry over almost exactly, but the statistical
significance does not, because daily large-cap rotation simply has less independent
information per unit time than a wide hourly crypto panel. This sharpens rather than
overturns the program's headline. The cross-sectional regime is genuinely the
stronger ML regime in both markets (it produces a real, costed, positive edge where
single-series direction forecasting produces noise), but whether that edge clears a
multiple-testing haircut depends on the breadth and rebalance frequency of the panel:
crypto clears it, daily large-cap equities fall just short. The honest multi-market
claim is "the edge is real and reproducible across asset classes, but its deflated
significance scales with panel breadth and turnover, and is decisive only in the
widest, fastest panel."

**Survivorship.** The equity universe is a fixed list of *current* large-cap members,
so names that were large-cap earlier but have since left the index (delistings,
takeovers, demotions) are absent. This is a mild upward bias on the long leg; the
result is survivorship-aware but not survivorship-free, and should be read as an
upper-ish bound rather than a clean estimate. Later listings enter the cross-section
only once they trade (within-live-span gating), so there is no look-ahead onto
pre-listing dates. A fully survivorship-free run on a point-in-time membership history
is the natural follow-up.

---

## 8. Trees versus neural networks on the identical cross-sectional task

The headline of this study is that gradient-boosted trees earn a real, deflation-
surviving cross-sectional edge. The fair counter-question is whether a neural network,
given the same panel, the same forward cross-sectional rank label, the same purged
walk-forward, the same rotation sizing search, and the same per-fill cost model, does
any better. The literature's repeated finding is that it does not: on tabular financial
cross-sections, shallow trees tie or beat deep nets more often than the reverse. We
tested this directly rather than asserting it.

Two neural families now carry real, costed, out-of-sample ledgers produced on a managed
96 GB accelerator (the full panel exceeds a local card): a per-bar feed-forward network
(mlp) scored over the pooled bar-by-pair cross-section, and a per-pair recurrent network
(lstm) over a causal lookback of each pair's own feature history. Both are scored from
their per-bar out-of-sample ledger on exactly the apparatus the trees used, including the
same False-Strategy-Theorem bar.

**Result.** Both neural families lose money net of costs on this task. The feed-forward
network posts an out-of-sample profit factor of about 0.98 across the horizons tested,
with a small negative annualised Sharpe; the recurrent network is weaker still, with a
profit factor below 1 and a clearly negative Sharpe. Neither comes close to the
multiple-testing deflation bar that all three tree families clear, and neither approaches
the trees' median out-of-sample profit factor of about 1.12. The full per-(family,
horizon) rows, with per-bar and annualised Sharpe, profit factor, and Deflated Sharpe,
are in `tables/family_dsr.md` (neural section) and `figures/fig5_nn_families.png`.

**Reading.** This is a clean, same-task confirmation of the literature's tabular finding
on our own data: trees beat the neural families on the cross-sectional rotation, under
identical costs and validation. The neural families are not a near-miss that better
tuning would rescue here; they are below break-even after costs, while the trees clear a
genuine deflation haircut. The remaining neural families (the gated-recurrent and
temporal-convolution sets and the cross-pair attention set) did not produce a usable
ledger on this pass; the attention model in particular has an unresolved degenerate-bar
issue at full panel width. They are reported as deferred, with no number attached, rather
than estimated.

---

## Files
- `scripts/run_cross_sectional_ml.py`: loads the banked cross-sectional and single-
  series results, recomputes DSR, PBO, and effective-N with the program's deflation
  toolkit, and writes the regime tables and the regime figures. Idempotent and robust
  to missing inputs.
- `scripts/run_family_dsr.py`: builds the family-level deflation table over the three
  completed tree families from the clean lgbm ledger plus the saved xgb and catboost
  out-of-sample values, computes the False-Strategy-Theorem bar and the rigorous
  lgbm per-horizon DSR, and writes the family table and the family figures. Light
  analysis only; no model training and no panel rebuild.
- `scripts/run_nn_families.py`: folds the neural families' real per-bar out-of-sample
  ledgers into the family deflation table. For each (neural family, horizon) with a
  non-empty ledger it recomputes per-bar and annualised Sharpe, profit factor, and the
  Deflated Sharpe against the same False-Strategy-Theorem bar as the trees, appends the
  rows to the family table, writes the neural figure, and records families with no
  usable ledger as deferred. Light analysis only; reads ledgers, trains nothing.
- `tables/family_dsr.{csv,md}`: per (family, horizon) OOS Sharpe, PF, and deflation
  verdict, plus the lgbm rigorous DSR detail and the neural-families section.
- `tables/nn_families_summary.json`: machine-readable neural-family rows plus the list
  of families folded versus deferred.
- `tables/xs_per_horizon.{csv,md}`: per-horizon OOS PF, Sharpe, and DSR (lgbm).
- `tables/dsr_by_regime.{csv,md}`: the single-series versus cross-sectional comparison.
- `tables/multimarket_xsection.{csv,md}`: FX and equity cross-sectional (bonus).
- `scripts/fetch_us_equity_daily.py`: downloads adjusted daily bars for the bounded
  large-cap US universe from the daily-OHLC data API. One process, paced well under
  the shared rate ceiling, resumable, tiny footprint.
- `scripts/run_us_equity_xsection.py`: builds the daily US-equity panel, reuses the
  cross-sectional engine unmodified (features, label, model, rotation, walk-forward),
  and writes the per-horizon out-of-sample profit factor, Sharpe, and Deflated Sharpe
  plus the crypto comparison. Single process, RAM-bounded.
- `tables/us_equity_xsection.{csv,md}`: per-horizon US-equity OOS PF, Sharpe, and DSR,
  with the crypto-versus-equities comparison and the survivorship caveat.
- `tables/us_equity_xsection_summary.json`: machine-readable US-equity result.
- `tables/summary.json`, `tables/family_dsr_summary.json`: machine-readable results.
- `figures/fig1_regime_dsr_pbo.png`, `figures/fig2_xs_per_horizon.png`,
  `figures/fig3_family_dsr.png`, `figures/fig4_regime_dsr_survival.png`,
  `figures/fig5_nn_families.png`, `figures/fig_us_equity_xsection.png`.
