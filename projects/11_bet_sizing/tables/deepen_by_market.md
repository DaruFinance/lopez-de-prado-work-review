# Project 11 — Bet Sizing DEEPENING (paired, by market)

Paired per-instrument deltas vs the FIXED-size book. ΔDSR>0 = prob/disc sizing improves the deflated headline; turn_ratio<1 = turnover (overtrading) collapses. frac_pos = fraction of instruments with Δ>0; sign_p = two-sided sign-test p-value (H0: prob/disc no better than fixed).

| market   |   n |   med(ΔDSR prob) |   frac_pos(ΔDSR prob) |   sign_p(ΔDSR prob) |   med(ΔDSR disc) |   frac_pos(ΔDSR disc) |   sign_p(ΔDSR disc) |   med turn_ratio prob |   med turn_ratio disc |   med pbo_prob |
|:---------|----:|-----------------:|----------------------:|--------------------:|-----------------:|----------------------:|--------------------:|----------------------:|----------------------:|---------------:|
| crypto   |  27 |          -0.0021 |                0.4444 |              0.7011 |           0.0015 |                0.5185 |              1      |                0.1965 |                0.1633 |         0.4762 |
| equities |   7 |          -0.0327 |                0.2857 |              0.4531 |           0.0363 |                0.7143 |              0.4531 |                0.1335 |                0.1287 |         0.4841 |
| forex    |   8 |          -0.0032 |                0.375  |              0.7266 |           0.0003 |                0.5    |              1      |                0.165  |                0.1432 |         0.621  |
| ALL      |  42 |          -0.0022 |                0.4048 |              0.28   |           0.0026 |                0.5476 |              0.644  |                0.1695 |                0.1557 |         0.4881 |


## PF deltas (cost-sensitivity proxy)

| market   |   med(ΔPF prob) |   frac_pos(ΔPF prob) |   sign_p(ΔPF prob) |   med(ΔPF disc) |   frac_pos(ΔPF disc) |   sign_p(ΔPF disc) |
|:---------|----------------:|---------------------:|-------------------:|----------------:|---------------------:|-------------------:|
| crypto   |         -0.0251 |               0.2593 |             0.0192 |          0.0066 |               0.5556 |             0.7011 |
| equities |          0.007  |               0.5714 |             1      |          0.1157 |               0.8571 |             0.125  |
| forex    |         -0.0217 |               0.5    |             1      |          0.0768 |               0.625  |             0.7266 |
| ALL      |         -0.0189 |               0.3571 |             0.0884 |          0.0167 |               0.619  |             0.1641 |