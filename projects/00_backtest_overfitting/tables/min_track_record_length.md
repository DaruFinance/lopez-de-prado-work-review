# Minimum Track Record Length (MinTRL) and Minimum Backtest Length (MinBTL) per market

MinTRL is computed against the multiple-testing null benchmark (the False-Strategy-Theorem expected maximum), the same bar the deflated test uses. MinBTL is the track length so a skill-less search of N trials would not be expected to produce a Sharpe as high as the one observed. Sharpe ratios are annualised for display; the underlying statistics are per observation (daily). Skew and kurtosis use the Gaussian reference (0, 3), matching the corpus-level deflated-Sharpe computation.

| market    | winning pair   |   N trials |   eff-N |   best SR (ann) |   null E[max] (ann) |   observed track (yrs) |   MinTRL (yrs) |   MinBTL nominal-N (yrs) |   MinBTL eff-N (yrs) | track >= MinBTL?   | best > null?   |
|:----------|:---------------|-----------:|--------:|----------------:|--------------------:|-----------------------:|---------------:|-------------------------:|---------------------:|:-------------------|:---------------|
| Crypto    | ALGO           |      50000 |     434 |           2.001 |               2.721 |                   2.48 |            inf |                     4.49 |                 2.26 | no                 | no             |
| US Equity | QQQ            |      22500 |      39 |           3.206 |              10.886 |                  19.35 |            inf |                     1.6  |                 0.46 | yes                | no             |
| Forex     | USDJPY         |      20000 |      26 |           1.779 |               7.701 |                   3.71 |            inf |                     5.12 |                 1.28 | no                 | no             |

Infinite MinTRL means the observed best Sharpe is at or below the multiple-testing null, so no finite track record makes it credible against that benchmark.
