# Sample Uniqueness and the Sequential Bootstrap

How overlapping labels make financial observations non-independent, what that
does to effective sample size, and whether correcting for it helps a model.

## The claim being tested

When you label a bar with a triple barrier (take-profit, stop-loss, or a
maximum-holding vertical barrier), the label of bar `t` is decided by the price
path over the whole interval `[t0, t1]` until the first barrier is touched. Two
labels whose intervals overlap therefore share information: they are partly
driven by the same future returns. The labels are not independent, and the usual
assumption behind bootstrapping, bagging, and effective-sample-size counting (one
row equals one independent observation) is false.

Lopez de Prado formalises this in three steps:

1. **Concurrency.** For each bar `t`, the concurrency `c_t` is the number of
   labels whose interval is live at `t`.
2. **Average uniqueness.** The average uniqueness of label `i` is the mean over
   its interval of `1 / c_t`. A label that is mostly alone has uniqueness near 1;
   a label that overlaps many others has uniqueness near 0.
3. **Effective sample size.** Summing the average uniqueness over all labels
   gives an effective number of observations that is far below the raw count.
   Counting rows overstates how much independent information you actually have.

He then proposes the **sequential bootstrap**: instead of drawing rows uniformly
with replacement, draw them one at a time with a probability that favours rows
whose interval overlaps least with the rows already drawn. The mechanical claim
is that a sequential-bootstrap sample has higher average uniqueness than a
standard IID bootstrap sample, so a bagged model built on sequential bootstraps
trains on less-redundant data.

We reproduce all of this on real bars across three markets, then ask the
question that actually matters: does the correction improve a model's
out-of-sample generalization, or just its bookkeeping?

## Data and method

Real data only, full available history per instrument:

- **Crypto:** 27 Binance USD-margined perpetual futures, 1-minute base bars
  aggregated to dollar bars.
- **US equities:** 7 liquid ETFs (SPY, QQQ, IWM, XLK, XLF, XLE, XLV), 1-minute
  regular-trading-hours base bars aggregated to dollar bars.
- **Forex:** 8 major pairs, 1-minute bars aggregated to tick bars (spot FX has
  no traded volume, so trade count is the information clock).

Each instrument is reduced to roughly 20,000 bars so the markets are compared on
a common footing. Events are sampled with a symmetric CUSUM filter (a label fires
whenever the cumulative absolute log-return since the last event crosses a
volatility-scaled threshold), which is the dense event sampler the chapter
assumes. Each event is labelled with a triple barrier using full intrabar
high/low (never close-only), so the first-touch interval `[t0, t1]` is the real
holding span. The binary label is whether the side-signed gross return at the
first touch was positive.

Everything is causal: the label of bar `t0` resolves forward in time but is
attributed to `t0` with its true interval, and every feature at `t0` uses only
information available at or before `t0`.

The model question is answered with bagged decision trees under purged k-fold
cross-validation with an embargo, so no training label can leak into a test
fold. We compare two ensembles:

- **Naive:** standard IID bagging, full-size bags, unweighted trees.
- **Corrected:** each bag drawn by the sequential bootstrap, bag size scaled to
  the mean average uniqueness (so we stop oversampling redundant rows), and
  trees weighted by return-attribution sample weights (each bar's return is
  shared across the labels concurrent with it, so overlapping labels do not each
  claim the same return).

We score out-of-sample log-loss and AUC, the in-sample-to-out-of-sample log-loss
gap (a direct overfit measure), and the cross-fold stability of feature
importances.

### Performance and verification

The concurrency scan, the average-uniqueness scan, the CUSUM filter, and the
sequential-bootstrap draw are all Numba-compiled. Each kernel is checked against
an independent pure-Python reference and is bit-identical:

| kernel | max absolute difference vs reference |
| --- | --- |
| concurrency | 0 (exact) |
| average uniqueness | 0.0 |
| sequential bootstrap (draw indices) | 0 (exact) |
| CUSUM events | 0 (exact) |
| return-attribution weights | 0.0 |

The sequential bootstrap uses an incremental formulation: rather than rescoring
every candidate on every draw (the textbook quadratic version), it keeps a
running concurrency vector and updates only the candidates whose intervals touch
the bars of the just-drawn label. The incremental kernel reproduces the slow
kernel's draw sequence exactly. We never build a dense bars-by-labels indicator
matrix; intervals are stored as integer spans and the per-bar candidate index is
a compact sparse structure, so peak memory for the full multi-market run stayed
under 3 GB. Two closed-form synthetic checks (disjoint unit intervals give
uniqueness exactly 1; N identical length-L intervals give uniqueness exactly
1/N) confirm the estimator itself is correct; these are used only to validate the
math, never as reported results.

## Results

### Overlap is severe and effective sample size collapses

Across every instrument in all three markets, mean average uniqueness sits
between 0.40 and 0.49. The information content of the labels is roughly 41 to 44
percent of what the raw row count implies.

| market | instruments | median labels N | median mean uniqueness | median effective N | effective-N ratio |
| --- | --- | --- | --- | --- | --- |
| crypto | 27 | 14,986 | 0.443 | 6,645 | 0.443 |
| equities | 7 | 14,742 | 0.415 | 6,114 | 0.415 |
| forex | 8 | 14,892 | 0.408 | 6,048 | 0.408 |

A naive workflow that treated 15,000 overlapping labels as 15,000 independent
observations would be assuming more than twice the information actually present.
The effect is slightly worse in equities and forex than in crypto, tracking their
marginally higher mean concurrency.

### Wider barriers make it worse

The overlap is driven by interval length. Widening the barriers lengthens the
holding intervals, raises concurrency, and pushes average uniqueness down
monotonically in every market. Whatever the label design, the lesson is the same:
the longer a label takes to resolve, the less independent it is.

### The sequential bootstrap raises sample uniqueness, as advertised

In all 42 instruments without exception, a sequential-bootstrap sample has higher
average uniqueness than a standard IID bootstrap sample. The median lift is about
+0.016 to +0.018 in uniqueness terms. The mechanical claim reproduces cleanly and
universally. The absolute size of the lift is modest here because at a mean
concurrency near 2.3 there is only so much redundancy to remove.

### Does it help the model? Honest answer: it trades discrimination for robustness

This is where the result is more interesting than the textbook.

- **Out-of-sample discrimination is slightly worse with the corrections.** Naive
  bagging had lower out-of-sample log-loss in all 42 instruments, and higher AUC
  on average (about 0.62 naive versus 0.56 corrected). Smaller, uniqueness-scaled,
  reweighted bags see less effective data, so the corrected ensemble captures a
  bit less signal.
- **But the overfit gap shrinks.** The in-sample-to-out-of-sample log-loss gap is
  smaller with the corrections in 40 of 42 instruments. The corrected ensemble
  generalises closer to how it trains, which is exactly what removing redundant,
  over-counted observations should do.
- **And feature importances get more stable.** The cross-fold rank correlation of
  feature importances is higher with the corrections in 34 of 42 instruments, so
  the model's explanation of itself is more reproducible.

The corrections do what the theory says at the level of the sample (uniqueness up,
overfitting down, importances steadier), but on this single-series, weak-signal
labelling problem that robustness comes at a small cost to raw out-of-sample
discrimination rather than a free improvement in it.

## Figures

- `fig1_avg_uniqueness_by_market.png` : distribution of average uniqueness per
  market; mass sits well below 1.
- `fig2_effective_N_vs_N.png` : effective sample size against the naive row count,
  with the ratio annotated.
- `fig3_seq_vs_std_bootstrap.png` : average uniqueness of sequential versus
  standard bootstrap samples, by market.
- `fig4_isoos_gap_by_market.png` : the in-sample-to-out-of-sample overfit gap and
  the out-of-sample log-loss, naive versus corrected.
- `fig5_feature_importance_stability.png` : cross-fold feature-importance
  stability, naive versus corrected.
- `fig6_overlap_vs_barrier_width.png` : how uniqueness falls as barriers widen and
  intervals lengthen.

## Tables

- `by_market_summary.csv` : per-market medians for every metric above.
- `per_instrument.csv` : full per-instrument detail.
- `uniqueness_by_horizon.csv` : uniqueness and concurrency as the barrier width
  is swept.
- `results.md` : the same content rendered as markdown.

## Verdict

The accounting is real and it reproduces everywhere: overlapping labels carry
under half the independent information their count suggests, and the sequential
bootstrap reliably raises sample uniqueness. The downstream model benefit is more
nuanced and worth stating plainly. The non-IID corrections reduce overfitting and
stabilise feature importances, but they do not improve out-of-sample
discrimination on this single-series direction-labelling problem; they slightly
reduce it. The corrections are best understood as a robustness and honesty
treatment, not a performance booster. This study is methodological: it quantifies
an information-accounting effect and its consequences, and it does not claim a
tradable edge.
