# Information-driven bars, clean 1m cross-section (27 Binance perps, 2022-2024, ~daily bars)

Median across pairs. Lower |skew|, excess kurtosis, |AC(1)|, JB, count-CV are better.

| bar_type   |   pairs |   med_abs_skew |   med_exkurt |   med_abs_ac1 |    med_jb |   frac_normal |   med_count_cv |
|:-----------|--------:|---------------:|-------------:|--------------:|----------:|--------------:|---------------:|
| time       |      27 |         0.3672 |       5.2924 |        0.0199 | 1279.47   |             0 |         0.0825 |
| tick       |      27 |         0.0931 |       1.6083 |        0.0266 |  110.891  |             0 |         0.7936 |
| volume     |      27 |         0.1463 |       1.4822 |        0.0257 |   96.8463 |             0 |         0.7357 |
| dollar     |      27 |         0.1592 |       1.544  |        0.0237 |  114.594  |             0 |         1.0221 |

## Count stability among information bars (median CV)

| bar_type   |   count_cv |
|:-----------|-----------:|
| tick       |     0.7936 |
| volume     |     0.7357 |
| dollar     |     1.0221 |