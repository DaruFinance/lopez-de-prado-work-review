# Meta-Labeling an *Edged* Primary: correcting "precision filter, not alpha"

**This study corrects sibling study 03 (the edgeless-primary meta-labeling
study).** The apparatus is the same: triple-barrier outcomes, a tree secondary,
**purged k-fold cross-validation**, realistic **per-fill costs**, and the
**Deflated Sharpe Ratio (DSR) as the headline metric**. The one thing that
changes is the only thing that mattered: the **primary now actually has an
edge**.

Headline up front, because the house rule is to say so plainly: **once Lopez de
Prado's stated precondition is met (meta-labeling is applied to a primary that
already has an edge), the secondary improves a real edge rather than rescuing a
bad one.** Profit factor rises from 1.26 to 1.79 and mean per-trade P&L rises
from 47 bp to 148 bp out-of-sample, net of costs, with precision lifting from
53.0% to 63.6%. The blanket conclusion of study 03 ("precision filter, not
alpha") does not survive once the precondition is honoured.

---

## 1. What LdP actually says (and the precondition study 03 broke)

Lopez de Prado's meta-labeling is a **two-layer** design:

- a **primary** model decides the **side** of every bet;
- a **secondary** ("meta") model decides **whether to act** and **how big**. It
  can veto and size, but it cannot flip the side.

The crucial, frequently-skipped sentence is the **precondition**: the primary is
assumed to **already have an edge** (high recall, mediocre precision). The
secondary's job is to *raise precision* by throwing away the primary's worst
bets and concentrating size into its best ones. Meta-labeling is therefore a
**multiplier on an existing edge, not a generator of edge**. If the primary's
bets carry no real edge net of costs, there is nothing for the filter to
concentrate.

> ### The claim being corrected
> **What LdP says:** apply meta-labeling on top of a primary model that already
> has an edge; the secondary lifts precision and sizes the bets.
> **What study 03 did:** meta-labeled a **vanilla EMA crossover**, a primary
> with **no established edge** net of costs on liquid markets.
> **What study 03 concluded:** "meta-labeling is a precision filter, not an alpha
> source... 0 of 42 instruments clear DSR > 0.95."
> **The flaw:** the precondition was **violated**. The secondary was asked to
> concentrate an edge that was not there. Garbage in: the null is about the
> *primary*, not about meta-labeling.

Study 03's own writeup actually anticipates this in its limitations (the primary
is intentionally vanilla; the negative deflation result is about the apparatus's
limits given a weak primary, not a claim that meta-labeling never helps) and
lists "a primary with genuine edge to begin with" as the first needle-mover.
**This study is that follow-up, and it inverts the headline.**

---

## 2. Method and data

| Component | This study (edged primary) | Study 03 (edgeless primary) |
|---|---|---|
| **Primary** | a **proven structural order-flow / open-interest edge** (a closed, pre-validated strategy, reused **read-only** through its verified engine) | vanilla EMA(fast)/EMA(slow) crossover |
| **Precondition** | **met** | **violated** |
| Side construction | the proven edge's exact, causal entry signal (direction set by the funding sign, gated by an open-interest growth z-score) | sign of (EMA_fast minus EMA_slow) |
| Labels | triple-barrier outcome of each primary entry (1 = the bet made money net of costs) | same |
| **Secondary** | bagged forest trees on **causal** features (OI z-score, realised volatility, momentum, funding sign, hour, side) | bagged trees on causal technical features |
| CV / leakage control | walk-forward windows; secondary trained only on in-sample entries whose triple-barrier label **fully resolves before the in-sample / out-of-sample boundary** (purged); knob selection in-sample only | purged k-fold (6 folds, 1% embargo) |
| Costs | the verified engine's **per-fill** crypto cost model (taker / maker), applied to every entry and exit | per-side bp on entry and exit, scaled by bet size |
| Bars | the engine's perp 30-minute grid | dollar / tick bars (about 20k) |
| **Headline metric** | **Deflated Sharpe Ratio**, plus PBO and effective-N | same |

- **Real data only.** No synthetic series; every fill is costed; every meta
  feature is causal (lagged one bar at load or built from already-lagged
  series); triple-barrier exits use full intra-bar high and low, never
  close-only.
- **Full per-trade ledger.** Both arms (primary-alone and meta-gated), both
  pairs, both in-sample and out-of-sample phases, are stored per trade. Every
  statistic below is computed off that one ledger, not re-run.
- **IS-tunable, not enumerated strategies.** The structural shape (the proven
  order-flow / open-interest primary, triple-barrier, tree secondary) is one
  strategy; the numeric knobs
  (threshold, stop, reward-to-risk, holding) are tuned in-sample per window.

---

## 3. The corrected result (OOS, net of per-fill costs)

| arm | trades | PF | per-trade bp | annual SR | **DSR** |
|---|---|---|---|---|---|
| primary (edge alone) | 166 | 1.26 | 47.3 | 10.3 | 0.636 |
| **meta (gate + size by p)** | 107 | **1.79** | **148.1** | **20.5** | **0.778** |

- **Profit factor: 1.26 to 1.79** (+0.53). The edge was already above 1;
  meta-labeling *amplifies* it instead of merely dragging a loser to
  break-even.
- **Per-trade P&L: 47.3 to 148.1 bp** (about 3.1 times). The secondary keeps the
  high-conviction subset, where the realised edge per bet is much larger.
- **Precision vs base rate: 53.0% to 63.6%.** The fraction of bets that are
  profitable rises by +10.6 pp, exactly the precision lift LdP describes, but now
  acting on real bets, so it shows up in P&L, not just in a confusion matrix.
- **The gate keeps 64% of the edge's bets** and vetoes the low-confidence 36%.
- **DSR (headline): 0.636 to 0.778.** Meta-labeling moves the deflated metric in
  the right direction on a primary that already has something to deflate. Study
  03's every-variant near-zero DSR is the signature of an *absent* primary edge.
- **Per-pair, the lift holds on both pairs**: pair A 1.32 to 1.49, pair B 1.23
  to 1.96 (see `by_pair.csv`).
- See the four figures: profit-factor and per-trade lift, precision vs base
  rate, DSR vs the 0.95 hurdle, and the out-of-sample equity curves.

> ### What changed
> **Study 03 said:** "meta-labeling is a precision filter, not an alpha source"
> (median meta PF about 1.0, 0 of 42 clear DSR).
> **With the precondition met, we find:** meta-labeling **improves a real edge**.
> PF 1.26 to 1.79, per-trade 47 to 148 bp, precision 53% to 64%, DSR 0.64 to
> 0.78.
> **Corrected reading:** "precision filter, not alpha" describes meta-labeling
> applied to a **non-edge**. Applied as LdP specifies, on top of an edge, the
> precision filter *is* the alpha amplifier. The original null was a statement
> about the primary, mislabeled as a statement about the method.

---

## 4. Honest caveats (the result is real but bounded)

- **DSR does not clear the publication bar on this sample.** This is a focused,
  two-pair demonstration (166 primary and 107 meta out-of-sample trades across
  walk-forward windows). The DSR rises to 0.78, which is a clear improvement, but
  it does **not** itself clear the program's **0.95** publication bar on this
  2-pair sample. The point of the study is the **direction and mechanism** of the
  correction (it inverts study 03's *blanket* claim), not a new "passes DSR"
  trophy. Scaling to the full set of proven edges is the natural next step and is
  left as future work.
- **DSR deflation basis.** Here DSR deflates the per-trade Sharpe against the
  dispersion of per-window Sharpes (the walk-forward windows as the trial set),
  which is a smaller trial count than study 03's 27-knob grid. The headline
  *comparison* (primary vs meta under the same deflation) is the load-bearing
  number; the absolute DSR level is conditioning-dependent and reported as such.
- **One edge, one cost regime.** The primary is a single proven crypto strategy
  under one per-fill cost model. The claim is existence (meta-labeling helps when
  the precondition holds), which is all that is needed to correct a *blanket*
  null; it is not a claim that it helps for every edge.
- **No look-ahead, costed, full ledger.** All the program guardrails hold; the
  knob selection and secondary fit touch only in-sample data, and the secondary's
  training labels are purged at the in-sample / out-of-sample boundary.

---

## 5. How to reproduce

```bash
python3 scripts/run_metalabel_real_primary.py                # banked reproduction (default, fast)
python3 scripts/run_metalabel_real_primary.py --from-banked  # explicit
```

Outputs: `tables/{primary_vs_meta.csv, by_pair.csv, results.md, raw_metrics.json}`
and the four figures. The default mode reads the banked full per-trade ledger of
the original "meta-gate a proven edge" experiment and recomputes every statistic.

---

## 6. Takeaway

Study 03's apparatus was sound; its primary was not. Meta-labeling is not "a
precision filter that fails to add alpha"; it is a **conditional amplifier**.
Fed a real edge, the precision filter *is* where the extra alpha comes from
(here, about 3 times the per-trade P&L and PF 1.26 to 1.79). The corrected
statement is the one LdP actually makes, with the precondition restored to the
front of the sentence: **meta-labeling improves an edge you already have; it
cannot manufacture one you do not.**
