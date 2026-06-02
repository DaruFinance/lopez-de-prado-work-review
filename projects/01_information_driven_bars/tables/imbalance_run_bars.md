# Imbalance and run bars vs the existing bar types (3 Binance perps, 1m base, ~daily target)

Median across instruments. Lower |skew|, excess kurtosis, |AC(1)|, Jarque-Bera are closer to IID-Gaussian; higher frac_normal is better. The four new rows (tib / vib / trb / vrb) are the advanced LdP bars; the first four rows are the study's existing bars rebuilt on the same instruments.

| bar_type   |   pairs |   med_n_bars |   med_abs_skew |   med_exkurt |   med_abs_ac1 |     med_jb |   frac_normal |
|:-----------|--------:|-------------:|---------------:|-------------:|--------------:|-----------:|--------------:|
| time       |       3 |         1096 |         0.0667 |       3.5633 |        0.0199 |   580.102  |             0 |
| tick       |       3 |         1096 |         0.3162 |       1.065  |        0.0285 |    70.0045 |             0 |
| volume     |       3 |         1096 |         0.2613 |       1.9396 |        0.0131 |   183.926  |             0 |
| dollar     |       3 |         1095 |         0.148  |       1.0707 |        0.0227 |    66.8186 |             0 |
| tib        |       3 |         3313 |         0.0094 |      11.3178 |        0.0291 | 16346.6    |             0 |
| vib        |       3 |         2482 |         0.632  |       6.457  |        0.0153 |  5470.92   |             0 |
| trb        |       3 |         1235 |         0.0473 |       3.8793 |        0.0288 |   777.379  |             0 |
| vrb        |       3 |         1831 |         0.2122 |       2.9041 |        0.015  |   669.813  |             0 |

## Reference: cross-section medians from the main study (27 perps)

|        |   med_exkurt (27-perp study) |
|:-------|-----------------------------:|
| time   |                         5.29 |
| tick   |                         1.61 |
| volume |                         1.48 |
| dollar |                         1.54 |

*tib = tick-imbalance, vib = volume-imbalance, trb = tick-run, vrb = volume-run.*
