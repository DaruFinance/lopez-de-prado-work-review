# Backtest overfitting, the Deflated Sharpe Ratio, and effective trials

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018); "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" (*Journal of Portfolio Management*, 2014); "The Probability of Backtest Overfitting" (*Journal of Computational Finance*, 2017).

This project is the gate the rest of the work hangs on. It takes a large corpus of real, fully costed strategies across three markets and asks whether the best out-of-sample result is more impressive than what a search of that size would produce by luck alone. It implements the False Strategy Theorem (the expected maximum Sharpe of a set of skill-less trials), the Deflated Sharpe Ratio that corrects an observed Sharpe for the number and correlation of trials, the Probability of Backtest Overfitting via combinatorially-symmetric cross-validation (CSCV/PBO), and an effective-number-of-trials count from the eigenvalue participation ratio of the returns. Strategy Sharpes are computed on net-of-cost daily PnL. The corpus spans roughly 92,500 strategies: about 50,000 crypto, 22,500 equity, 20,000 FX.

Verdict: the keystone gate reproduces. The best out-of-sample Sharpe in each market sits below the skill-less expected maximum for a search that size (crypto 2.00 vs 2.72, equity 3.21 vs 10.89, FX 1.78 vs 7.70). The Deflated Sharpe is near zero in every market. Nothing here is deployable.

## Run

From the repo root. The full three-market run reads the per-strategy daily PnL corpus from the `LDP_PNL_DAILY` root and loops over crypto, equity, and FX:

```
bash projects/00_backtest_overfitting/run_full.sh
```

Single market (set `LDP_MARKET` to `crypto`, `equity`, or `fx`):

```
LDP_MARKET=crypto python3 projects/00_backtest_overfitting/scripts/run_corpus_overfit.py
```

The grid-search variant that builds the dual-moving-average trials directly from price bars (dollar bars for crypto and equities, tick bars for FX):

```
python3 projects/00_backtest_overfitting/scripts/run_overfit_at_scale.py
```

Cross-market summary (reads the per-market corpus CSVs the harness already wrote, no heavy recompute) and the Minimum Track Record / Minimum Backtest Length add-on:

```
python3 projects/00_backtest_overfitting/scripts/build_cross_market.py
python3 projects/00_backtest_overfitting/scripts/run_min_track_record.py
```

Data roots come from `config.py` and the `LDP_*` environment variables it reads. The corpus run needs `LDP_PNL_DAILY`; the grid-search run needs `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M`. See `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
