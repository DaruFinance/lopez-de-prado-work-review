# Information-driven bars, Crypto vs US Equities (median across instruments)

Crypto: 27 Binance perps (1m, 2022-2024). Equities: 9 Algoseek ETFs (1m). Matched ~daily bars. Lower excess kurtosis / |skew| / |AC(1)| is better.

|                      |   n |   med_exkurt |   med_abs_skew |   med_abs_ac1 |   frac_normal |
|:---------------------|----:|-------------:|---------------:|--------------:|--------------:|
| ('crypto', 'time')   |  27 |       5.2924 |         0.3672 |        0.0199 |             0 |
| ('crypto', 'tick')   |  27 |       1.6083 |         0.0931 |        0.0266 |             0 |
| ('crypto', 'volume') |  27 |       1.4822 |         0.1463 |        0.0257 |             0 |
| ('crypto', 'dollar') |  27 |       1.544  |         0.1592 |        0.0237 |             0 |
| ('equity', 'time')   |   9 |      18.3125 |         0.3484 |        0.0144 |             0 |
| ('equity', 'tick')   |   9 |      19.4676 |         0.3113 |        0.009  |             0 |
| ('equity', 'volume') |   9 |      12.9004 |         0.2973 |        0.0135 |             0 |
| ('equity', 'dollar') |   9 |      34.422  |         0.6211 |        0.0123 |             0 |