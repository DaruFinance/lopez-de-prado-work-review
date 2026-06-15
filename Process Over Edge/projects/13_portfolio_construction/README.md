# Hierarchical Risk Parity, Nested Clustered Optimization, and covariance denoising

López de Prado, *Advances in Financial Machine Learning*, Ch.16 (HRP); *Machine Learning for Asset Managers* (MP denoising, detoning, NCO); López de Prado & Lewis, "Detoning" / TIC (2018); and "Building Diversified Portfolios that Outperform Out of Sample".

This study compares portfolio constructors out of sample across two universes and all three markets (crypto, US equities, forex). It runs a walk-forward: weights are estimated in-sample on a 252-day window, held over the following 63-day out-of-sample window, and rolled forward by 63 days, so weights are never scored on the data that built them. The constructors are raw Markowitz, inverse-variance, equal weight (1/N), Hierarchical Risk Parity, and Nested Clustered Optimization, with Marchenko-Pastur covariance denoising and detoning applied where the recipe calls for them. The first universe is roughly 40 instruments scored daily close-to-close with a one-level taxonomy for TIC; the second is 300 de-correlated strategies per market over full daily-PnL history. The headline metric is realized out-of-sample portfolio variance (annualized vol) plus the Deflated Sharpe Ratio, with concentration (HHI / effective-N) and the covariance condition number reported alongside. The HRP recursive-bisection kernel is verified bit-identical to a NumPy reference.

## Verdict

Reproduced. HRP and NCO beat raw Markowitz on out-of-sample variance in the worst case 39 of 39, and tie 1/N on Sharpe. The denoising benefit is a function of q = T/N, the ratio of the sample length to the number of assets.

## Run

From the repository root:

```
# universe of ~40 instruments, TIC enabled
python3 projects/13_portfolio_construction/scripts/run_portfolio.py --universe assets \
    --is-win 252 --oos-win 63 --step 63 --tag assets_full

# 300 de-correlated strategies per market, all markets
python3 projects/13_portfolio_construction/scripts/run_portfolio.py --universe strategies --n-strat 300 \
    --is-win 252 --oos-win 63 --step 63 --tag strategies_full
```

`run_portfolio.py --selftest` runs the HRP bit-identity gate; `--smoke` does a short run; `--profile` profiles it. The wrapper `projects/13_portfolio_construction/run_full.sh` runs the selftest and both universes in sequence and caches the asset return panel so reruns are fast.

Data roots needed: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, `LDP_FX_1M`, `LDP_CRYPTO_1H`, `LDP_CRYPTO_DELISTED_1H` for the asset universe, and `LDP_PNL_DAILY` for the strategy universe. All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
