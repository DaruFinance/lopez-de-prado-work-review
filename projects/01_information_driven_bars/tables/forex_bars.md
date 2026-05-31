# Forex: time vs tick bars (8 majors, median)

Spot FX has no volume; tick count is the information clock. Gap-aware returns (weekend/rollover dropped). Lower exkurt is better.

| pair   |   exkurt_time |   exkurt_tick |   absskew_time |   absskew_tick |   ac1_time |   ac1_tick |   n_time |   n_tick |
|:-------|--------------:|--------------:|---------------:|---------------:|-----------:|-----------:|---------:|---------:|
| AUDUSD |         2.391 |         0.277 |          0.009 |          0.148 |      0.027 |      0.013 |     1092 |     1093 |
| EURGBP |         3.887 |         2.608 |          0.389 |          0.321 |      0.058 |      0.003 |     1093 |     1092 |
| EURUSD |         1.916 |         0.294 |          0.259 |          0.042 |      0.003 |      0.032 |     1092 |     1093 |
| GBPUSD |         8.078 |         1.718 |          0.018 |          0.063 |      0.022 |      0.005 |     1093 |     1092 |
| NZDUSD |         1.529 |         0.591 |          0.138 |          0.113 |      0.041 |      0.044 |     1093 |     1093 |
| USDCAD |         2.054 |         1.088 |          0.074 |          0.103 |      0.058 |      0.026 |     1093 |     1093 |
| USDCHF |         4.329 |         1.032 |          0.548 |          0.253 |      0.02  |      0.024 |     1092 |     1093 |
| USDJPY |         3.575 |         2.163 |          0.516 |          0.521 |      0.009 |      0.049 |     1092 |     1093 |

**Medians:** time exkurt 2.98 vs tick exkurt 1.06; |skew| 0.199 vs 0.130
