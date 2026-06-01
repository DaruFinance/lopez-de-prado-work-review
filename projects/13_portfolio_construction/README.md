# Project 13, Portfolio Construction: denoising/detoning, HRP, NCO, TIC vs Markowitz

Reproduces and stress-tests López de Prado's portfolio-construction toolkit on **real,
multi-market** data, **walk-forward**.

- **AFML Ch.16**, Hierarchical Risk Parity (tree clustering → quasi-diagonalisation →
  recursive bisection; *no matrix inversion*).
- **ML4AM Ch.2/4/7**, Marčenko-Pastur covariance **denoising** (constant-residual
  eigenvalue), **detoning** (remove the market eigenvector), and **Nested Clustered
  Optimization (NCO)**.
- **LdP & Lewis (2018)**, **Theory-Implied Correlation (TIC)** from a one-level asset
  taxonomy (assets universe only).
- **Controls**, mean-variance / min-variance on the **raw** sample covariance (the
  *Markowitz curse*), inverse-variance, and naive **1/N**.

## What is measured (walk-forward, causal)
Weights are estimated on a 252-day **in-sample** window and **held** over the following
63-day **out-of-sample** window, rolled by 63 days. Weights are never scored on the data
that built them. Per allocator we report:

- **OOS portfolio variance** (annualised vol), the headline.
- **DSR** (Deflated Sharpe Ratio, `lib/overfit.py`) of the realised OOS return stream,
  deflated against the menu of allocators tried.
- **Concentration**: HHI and effective-N of the weights.
- **Condition number** of the covariance actually fed to the allocator (exposes the
  Markowitz curse and how much denoising tames it).

## Two universes (different experiments)
1. **Across assets**, daily close-to-close returns of ~40 instruments: crypto perps
   (1m → daily), equity ETFs (Algoseek 1-min → daily), FX majors (1m → daily).
2. **Across strategies** (the LdP use-case), ~300 greedily de-correlated per-strategy
   daily-PnL series **per market**, allocate the risk budget across them.

## Run
```bash
bash run_full.sh                          # full walk-forward, both universes, all markets
# vol-targeted (clean comparable Sharpe; --vol-target = per-leg annualised vol, causal IS-scaler):
scripts/run_portfolio.py --universe assets     --vol-target 0.10 --tag assets_voltgt
scripts/run_portfolio.py --universe strategies --n-strat 300 --vol-target 0.10 --tag strategies_voltgt_n300
scripts/run_portfolio.py --universe strategies --n-strat 150 --vol-target 0.10 --tag strategies_voltgt_n150  # q=T/N>1: denoising active
python3 scripts/make_figures.py           # figures from the tables (no re-run)
# smoke (1-core, tiny):
scripts/run_portfolio.py --universe assets     --smoke --is-win 150 --oos-win 30 --step 30
scripts/run_portfolio.py --selftest            # Numba HRP bit-identity gate
```

See `writeup/README.md` for the full 8-section writeup, headline numbers, and verdict.

## Performance / RAM
Profiled on 1 core: **~98% of wall time is data loading** (ETF gzip-CSV + parquet), not
the portfolio math, the eigendecomposition + sklearn linkage are C/Fortran on a bounded
(≤300-asset) universe. The only pure-Python hot spot, the **HRP recursive bisection**, is
a Numba `njit` kernel verified **bit-identical** to the numpy reference (max |Δw| ≤ 5.6e-17).
The assembled asset daily-return panel is cached to `data_cache/p13/`, so asset reruns are
instant (56 s → 0.6 s). Full run ≈ 15-18 min wall, peak RSS ≈ 2.6 GB (sequential per-market,
freed between markets); covariance is N×N with N ≤ 300 (≈0.7 MB), nothing is tiled.

## Layout
```
13_portfolio_construction/
  run_full.sh                 exact full command + header (runtime/RAM/perf)
  scripts/run_portfolio.py    --universe {assets,strategies} --smoke --profile --selftest
  tables/                     portfolio_<tag>.csv  (one row per allocator per market)
  figures/  writeup/          (figures + writeup pass, after the full run)
```
