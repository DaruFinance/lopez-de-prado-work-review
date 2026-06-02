# Project 08: order-flow inputs, stationary transform vs raw level

Same ML task / purged CV / cost. Crypto only (true buyer-seller volume). RAW = cumulative signed-flow level (+ its lag). STATIONARY = order-flow imbalance + FFD of that level at the minimal d (per-fold, training-span ADF) that retains memory.

| instrument   |   n_obs |   d_used |   auc_raw |   auc_stat |   auc_gap |   sr_raw |   sr_stat |   sr_gap |   bp_raw |   bp_stat |   dsr_raw |   dsr_stat |
|:-------------|--------:|---------:|----------:|-----------:|----------:|---------:|----------:|---------:|---------:|----------:|----------:|-----------:|
| BTCUSDT      |    8759 |     0.4  |    0.4955 |     0.4922 |   -0.0033 |  -0.0271 |   -0.0656 |  -0.0385 |  -2.6779 |   -6.4454 |  nan      |        nan |
| ETHUSDT      |    8759 |     0.55 |    0.4942 |     0.5034 |    0.0092 |  -0.0364 |   -0.0551 |  -0.0187 |  -4.4871 |   -6.7142 |  nan      |        nan |
| SOLUSDT      |    8759 |     0.45 |    0.5025 |     0.4921 |   -0.0103 |  -0.0004 |   -0.0339 |  -0.0334 |  -0.0905 |   -6.7241 |  nan      |        nan |
| BNBUSDT      |    8759 |     0.55 |    0.4896 |     0.5028 |    0.0132 |  -0.0093 |   -0.0314 |  -0.0221 |  -1.0918 |   -3.6851 |  nan      |        nan |
| XRPUSDT      |    8754 |     0.4  |    0.5053 |     0.5004 |   -0.0049 |   0.0061 |   -0.0331 |  -0.0392 |   0.9421 |   -5.1605 |  nan      |        nan |
| LTCUSDT      |    8759 |     0.55 |    0.5041 |     0.4874 |   -0.0168 |   0.0003 |   -0.0469 |  -0.0472 |   0.043  |   -6.9627 |  nan      |        nan |
| POOLED       |   52549 |     0.5  |    0.4985 |     0.4964 |   -0.0021 |  -0.0111 |   -0.0443 |  -0.0332 |  -1.227  |   -5.9487 |    0.0175 |          0 |

- Pooled DSR (best costed fold-Sharpe, deflated): raw 0.017 (best SR +0.041, E[max] 0.064, n_trials 36); stationary 0.000 (best SR +0.012, E[max] 0.064, n_trials 36).
