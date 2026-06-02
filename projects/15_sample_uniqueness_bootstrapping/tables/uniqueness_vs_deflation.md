# Sample uniqueness vs the deflation penalty

Per-observation OOS Sharpe held fixed at `SR = 0.025`; selection trials `N_trials = 100`, trial-Sharpe variance `6.4e-05`, Gaussian returns (skew 0.0, kurtosis 3.0). Only the sample size moves between the nominal and effective columns.

PSR is the probability the true Sharpe beats zero. DSR is the PSR against the expected-max-Sharpe benchmark implied by the trials. The z-inflation factor is how much the PSR test statistic is overstated by using the nominal row count instead of the honest effective count.

| market | N | effective N | eff ratio | Sharpe SE (nominal) | Sharpe SE (effective) | SE inflation | PSR (nominal) | PSR (effective) | PSR drop | DSR (nominal) | DSR (effective) | DSR drop | z-inflation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto | 14,986 | 6,639 | 0.443 | 0.00817 | 0.01228 | 1.503 | 0.9989 | 0.9792 | 0.0197 | 0.7197 | 0.6508 | 0.0690 | 1.503 |
| equities | 14,742 | 6,111 | 0.414 | 0.00824 | 0.01280 | 1.553 | 0.9988 | 0.9746 | 0.0242 | 0.7181 | 0.6449 | 0.0732 | 1.553 |
| forex | 14,892 | 6,070 | 0.408 | 0.00820 | 0.01284 | 1.566 | 0.9989 | 0.9743 | 0.0246 | 0.7191 | 0.6445 | 0.0746 | 1.566 |
| all | 14,922 | 6,584 | 0.441 | 0.00819 | 0.01233 | 1.505 | 0.9989 | 0.9787 | 0.0201 | 0.7193 | 0.6502 | 0.0691 | 1.505 |

Minimum track record length to reach 95 percent PSR at this Sharpe is 4331 independent observations. It is a required count of independent observations, so it does not change with the sample you happen to hold. The point is the comparison: a nominal sample of about 15,000 overlapping labels supplies only about 6,000 to 6,700 independent observations, so a track that looks long enough on a row count can fall short on an honest count.
