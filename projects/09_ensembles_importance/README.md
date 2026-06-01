# Project 09, Ensembles & Feature Importance

Bagging-vs-boosting with hyper-parameter tuning (AFML Ch.6 & 9) and feature
importance MDI/MDA/clustered-MDA (AFML Ch.8 / ML4AM Ch.6), on a triple-barrier
labeled ML task across **crypto + US equities + forex**, causal-only, costed,
with **purged CV / CPCV** and the **Deflated Sharpe Ratio** as the headline metric.

## Two questions

1. **Bagging vs boosting.** A RandomForest-style bagged ensemble (low
   `max_features`, `min_weight_fraction_leaf > 0`, uniqueness sample-weights,
   `max_samples = avg label uniqueness`, the AFML 4.5/6.2/6.3 recipe) versus
   HistGradientBoosting. Hyper-tuned over a **purged** grid by **negative
   log-loss** (headline, AFML 9.4) and by **accuracy** (control). We compare the
   resulting bet's **OOS DSR** and the **IS−OOS log-loss overfit gap**.
2. **Feature importance.** MDI (in-sample tree impurity, the biased baseline) vs
   MDA (out-of-fold permutation, **log-loss scored**) vs **clustered-MDA**
   (correlated features clustered, whole clusters permuted). We report MDI's
   substitution-bias signal and the **stability of the top-feature set across
   CPCV paths** (Jaccard overlap).

## Layout

- `scripts/run_ensembles_importance.py`, idempotent driver (`--smoke` / `--profile`
  / `--verify` / `--jobs N`). Reuses `projects/03_meta_labeling/scripts/tbm.py`
  (triple-barrier labels + causal features) and `lib/overfit.py` (purged k-fold,
  CPCV, DSR). Shared `lib/{bars,overfit,fracdiff,style}.py` imported, not edited.
- `scripts/deepen.py`, deepening driver (`--jobs N`). Recomputes the
  40-instrument panel storing extra fields the headline run did not: clustered-MDA
  bias, three-way (MDI/MDA/clustered-MDA) selection stability + a random-Jaccard
  baseline, IS/OOS log-loss levels, and NLL-vs-ACC config agreement. Reuses the
  driver's verified internals read-only. BLAS pinned to 1 thread per process.
- `run_full.sh`, exact full command + sizing/RAM/runtime header.
- `tables/`, `figures/`, outputs (`per_instrument.csv`, `by_market_summary.csv`,
  `results.md`, `raw_results.parquet`; `fig1..fig4`; deepening:
  `deepen_per_instrument.csv`, `deepen_results.parquet`, `deepen_summary.md`,
  `fig5..fig7`).
- `writeup/WRITEUP.md`, the complete 8-section writeup (headline numbers, deepening,
  honest DSR verdict, paper-worthiness).

## Method notes

- **Labeled task** (fixed structural shape): EMA(20/60) crossover primary side →
  triple-barrier (pt=1.5σ, sl=1.0σ, max_hold=50) realised outcome → meta-label
  `y = 1[net P&L > 0]`, net of per-side cost (crypto 7 bp / equities 2 bp /
  forex 1 bp, entry+exit). Causal features only (vol, ma_gap, momentum 3/6/12,
  RSI, vol_ratio, OFI, range/ATR).
- **Purging.** `label_span` (event space) = `ceil(max_hold / median event gap)`;
  fed to `purged_kfold_splits` / `cpcv_splits` so train labels overlapping a test
  fold (plus embargo) are dropped.
- **DSR trials.** The pooled RF+HGB hyper-grid bet Sharpes are the multiple-testing
  trial set deflating the selected model's SR.

## Numba

Only the **non-sklearn** hot loops are JIT'd (sklearn fits are left to sklearn,
which is ~89% of runtime per the profile): the sequential-bootstrap draw
(AFML 4.5.2) and the co-event count + average label uniqueness (AFML 4.4). Both
are verified **bit-identical** against independent pure-Python references via
`--verify` (max|Δ|: uniqueness 5.6e-16, count 0, bootstrap index 0).

## Reproduce

```bash
python3 scripts/run_ensembles_importance.py --verify   # numba parity
python3 scripts/run_ensembles_importance.py --smoke    # 3-instrument 1-core sanity (~30 s)
python3 scripts/run_ensembles_importance.py --profile  # cProfile one instrument
./run_full.sh                                          # full run (~8-15 min, --jobs 16)
```

Determinism: `random_state=0`; per-process BLAS/OMP pinned to 1 thread so results
are independent of `--jobs`.
