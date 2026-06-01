# Cross-market backtest-overfitting at scale (~50k crypto + ~22.5k equity + ~20k forex strategies)

All three corpora are real, causal and realistically costed (crypto: fee/slip/funding; equity & forex via `lib/realism`: time-of-day spreads, commission, FX rollover/triple-Wednesday swap + weekend force-flat, equity short borrow).

| market    |   n_strat |   n_pairs |   best_SR_ann |   null_E[max]_ann |   DSR_best |   median_PBO |   eff_N_pooled | best<null?   | DSR>0.95?   |
|:----------|----------:|----------:|--------------:|------------------:|-----------:|-------------:|---------------:|:-------------|:------------|
| Crypto    |     50000 |        20 |         2.001 |             2.721 |      0.029 |        0.278 |            434 | yes          | no          |
| US Equity |     22500 |         9 |         3.206 |            10.886 |      0     |        0     |             39 | yes          | no          |
| Forex     |     20000 |         8 |         1.779 |             7.701 |      0     |        0.005 |             26 | yes          | no          |

*best_SR / null are annualised; DSR uses nominal N with the empirical trial-Sharpe dispersion (conservative). `best<null?` = is the single best of the whole corpus below the False-Strategy-Theorem expected max? `DSR>0.95?` = does it clear the deflated-significance bar?*
