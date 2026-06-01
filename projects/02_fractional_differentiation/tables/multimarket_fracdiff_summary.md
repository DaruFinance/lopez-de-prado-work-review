# Fractional Differentiation, multi-market summary (1h, FFD tau=1e-5)

Same d grid [0,1] step 0.05, same ADF logic, same memory metric (corr of FFD series with the log-price level) across all markets.


| market | n | median d* | IQR (Q1-Q3) | % d*<1 | median corr@d* | median |corr|@d=1 |
|---|---|---|---|---|---|---|
| crypto | 505 | 0.150 | 0.100-0.150 | 100% | 0.981 | 0.010 |
| equities | 9 | 0.150 | 0.000-0.150 | 100% | 0.997 | 0.007 |
| forex | 3 | 0.100 | 0.050-0.125 | 100% | 0.994 | 0.010 |

LdP reference band for equities/FX: d* ~ 0.3-0.6.


## Per-instrument (equities + forex)

| market   | instrument   |   d_star |   adf_stat |   adf_p |   corr_level_dstar |   corr_level_d1 |
|:---------|:-------------|---------:|-----------:|--------:|-------------------:|----------------:|
| equities | SPY          |     0.15 |    -4.1216 |  0.0009 |             0.9968 |          0.0082 |
| equities | QQQ          |     0.25 |    -4.0704 |  0.0011 |             0.9941 |          0.0052 |
| equities | IWM          |     0.15 |    -4.0022 |  0.0014 |             0.9917 |          0.0066 |
| equities | XLE          |     0    |    -2.956  |  0.0392 |             1      |          0.012  |
| equities | XLF          |     0.1  |    -3.0181 |  0.0332 |             0.9946 |          0.0075 |
| equities | XLK          |     0.15 |    -2.9493 |  0.0399 |             0.9958 |          0.0031 |
| equities | XLV          |     0.15 |    -3      |  0.0349 |             0.9969 |          0.0024 |
| equities | VXX          |     0    |    -5.8801 |  0      |             1      |          0.0194 |
| equities | UVXY         |     0    |    -5.997  |  0      |             1      |          0.0256 |
| forex    | EURUSD       |     0.1  |    -3.6231 |  0.0053 |             0.9798 |          0.0096 |
| forex    | EURGBP       |     0    |    -4.5493 |  0.0002 |             1      |          0.0118 |
| forex    | USDJPY       |     0.15 |    -3.0387 |  0.0314 |             0.9943 |          0.0057 |