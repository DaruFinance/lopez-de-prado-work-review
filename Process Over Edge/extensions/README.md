# Extensions: a reflexive stress-test

The core repository argues that disciplined process beats the hunt for a single
edge. These extensions turn that argument back on itself. Each one is a genuine
attempt to improve a core method or to mine a new edge, and each is held to the
same bar the rest of the work uses.

Eighteen extensions were run. Every one had its acceptance bar written down
before any deep computation, under a firewall that separates discovery from
confirmation and an independent adversarial review of the results. None of them
manufactured an edge. A handful cleared the discovery pass, and every one of
those was overturned on closer inspection. The point is not the box score. It is
that the same procedure catches your own attempts as readily as anyone else's.

## What each folder tests, and how it landed

- **S01_overfit_tooling** — overfitting and deflated-Sharpe tooling checks.
  Reproduced as tooling; no edge.
- **S02_hybrid_bars** — hybrid information-bar construction. No edge beyond the
  bar-Gaussianizing effect already established in the core work.
- **S03_ffd_vs_shortcuts** — fractional differencing against cheaper shortcuts.
  A tautology: the minimum differencing order maximizes retained memory given
  stationarity by construction, and there is no downstream gain.
- **S04_metalabel** — a cross-validation-chosen meta-label cutoff. The claimed
  "variance reduction" is a sample-size tautology; once normalized through the
  central limit theorem it shows no improvement (p = 0.087).
- **S05_vol_features** — longer-horizon volatility features. A label bug made the
  "10-bar-ahead" target 90% past data; corrected, the method wins at zero
  horizons, and a trivial AR(1) of past realized volatility beats the
  microstructure features.
- **S06_maxfeatures** — a sweep over the `max_features` mechanism. The underlying
  task carried zero signal (every instrument worse than a coin flip), so the
  sweep was vacuous.
- **S07_spread_breadth** — cointegration-spread mean-reversion on
  crypto-perpetual pairs, built leak-free at breadth. It cleared the
  pair-selection-permutation deflation null (p = 0.002 over 15,360 effective
  trials), so it is statistically real, but it failed the robustness bar:
  ex-worst-window reward/risk of 0.27 and a cohort sign-split, with one token
  group net-positive and the other net-negative. Demoted to an honest
  characterization, not a tradeable edge.
- **S08_shrink_optimize** — shrink-then-optimize portfolios. Stale zero-filled
  data faked low volatility; on clean folds, hierarchical risk parity still wins
  (p = 0.75 against the random-matrix-theory alternative).
- **S09_causal_robust** — short-reversal robustness. Unclustered standard errors
  on a correlated panel inflated the statistic to t = -11.9; date-clustered, it
  is about t = -1.5 and untradeable (sleeve Sharpe about -0.7).
- **S10_uniqueness_lossweight** — sample-uniqueness loss weighting. No
  out-of-sample improvement; an honesty correction, not an edge.
- **S11_regime_deploy** — regime-gated deployment. No deflation-survivable edge.
- **S13_precision_gate** — precision-gating a primary signal. No deflated edge on
  the tested sample.
- **S2_FUNDING_CARRY** — a funding-carry sleeve. The premium is real and
  persistent, but it is not tradeable once the price residual and costs are
  charged against it.
- **SURVIVOR_SKELETON** — the shared scaffold reused by the breadth tests.

## How to run

From the repository root:

```
python3 extensions/<Sxx_dir>/<script>.py
```

Each script resolves the shared library and its data roots through the
repository's central `config.py`, which reads each root from an `LDP_*`
environment variable and falls back to a repo-relative default. Point the
relevant variables at your own copies of the data before running. The variable
names and the sources they expect are listed in `../DATA.md` and `../config.py`.
Run outputs are written into each extension's own folder.
