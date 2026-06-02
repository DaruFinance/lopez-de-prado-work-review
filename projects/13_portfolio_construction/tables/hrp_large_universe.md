# HRP/NCO scaling on a large crypto universe (walk-forward OOS)

OOS annualised variance and Sharpe per universe size N. The HRP-vs-1/N
variance ratio < 1 means HRP delivered lower out-of-sample variance than
equal-weight; smaller is a bigger HRP advantage.

| N | folds | HRP var | NCO var | Markowitz var | 1/N var | HRP/1N var | HRP/1N vol | NCO/1N var | HRP Sharpe | 1/N Sharpe | HRP effN | HRP turnover | 1/N turnover |
|---|------:|--------:|--------:|--------------:|--------:|-----------:|-----------:|-----------:|-----------:|-----------:|---------:|-------------:|-------------:|
| 25 | 24 | 0.51804 | 0.44254 | 0.4078 | 0.59737 | 0.8672 | 0.9312 | 0.7408 | 0.803 | 0.639 | 17.1 | 0.139 | 0.0 |
| 50 | 24 | 0.59431 | 0.55456 | 0.54052 | 0.65297 | 0.9102 | 0.954 | 0.8493 | 0.825 | 0.598 | 30.6 | 0.1697 | 0.007 |
| 100 | 24 | 0.33676 | 0.28751 | 0.36921 | 0.67611 | 0.4981 | 0.7057 | 0.4252 | 0.866 | 0.579 | 34.9 | 0.2986 | 0.0356 |
| 200 | 24 | 0.30739 | 0.33151 | 0.3052 | 0.6847 | 0.4489 | 0.67 | 0.4842 | 0.275 | 0.534 | 4.2 | 0.446 | 0.0637 |
| all | 24 | 0.32082 | 0.42988 | 0.45935 | 0.66872 | 0.4798 | 0.6926 | 0.6428 | 0.19 | 0.522 | 4.8 | 0.4907 | 0.1122 |
