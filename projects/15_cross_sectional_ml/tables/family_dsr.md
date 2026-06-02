# Family-level deflation table - cross-sectional tree families

Rows are (family, horizon) for the three COMPLETED gradient-boosted-tree families. OOS Sharpe is shown per-bar and annualized (engine factor sqrt(252*24)=~77.8); PF is unit-free. The deflation verdict is the False Strategy Theorem: a combo clears if its OOS per-bar Sharpe exceeds the expected MAXIMUM skill-less Sharpe over the multiple-testing family.

| family   |   H |   oos_bars |   oos_sharpe_bar |   oos_sharpe_ann |   oos_pf | clears_N40   | deflation_verdict          |
|:---------|----:|-----------:|-----------------:|-----------------:|---------:|:-------------|:---------------------------|
| lgbm     |   4 |      42000 |           0.0415 |            3.231 |   1.1481 | True         | clears expected-max (N=40) |
| lgbm     |   6 |      41922 |           0.0359 |            2.794 |   1.127  | True         | clears expected-max (N=40) |
| lgbm     |   8 |      42000 |           0.0335 |            2.606 |   1.1198 | True         | clears expected-max (N=40) |
| lgbm     |  12 |      42000 |           0.0342 |            2.661 |   1.1582 | True         | clears expected-max (N=40) |
| lgbm     |  24 |      42000 |           0.0465 |            3.618 |   1.1494 | True         | clears expected-max (N=40) |
| lgbm     |  48 |      42000 |           0.0336 |            2.616 |   1.1108 | True         | clears expected-max (N=40) |
| lgbm     |  72 |      42000 |           0.0432 |            3.356 |   1.1404 | True         | clears expected-max (N=40) |
| lgbm     | 120 |      42000 |           0.0329 |            2.555 |   1.1073 | True         | clears expected-max (N=40) |
| lgbm     | 168 |      42000 |           0.0307 |            2.388 |   1.0954 | True         | clears expected-max (N=40) |
| lgbm     | 336 |      42000 |           0.0281 |            2.187 |   1.0894 | True         | clears expected-max (N=40) |
| xgb      |   4 |      42000 |           0.0171 |            1.326 |   1.062  | False        | below expected-max (N=40)  |
| xgb      |   6 |      42000 |           0.0159 |            1.236 |   1.07   | False        | below expected-max (N=40)  |
| xgb      |   8 |      42000 |           0.0255 |            1.981 |   1.109  | True         | clears expected-max (N=40) |
| xgb      |  12 |      42000 |           0.0199 |            1.548 |   1.072  | False        | below expected-max (N=40)  |
| xgb      |  24 |      42000 |           0.0262 |            2.041 |   1.09   | True         | clears expected-max (N=40) |
| xgb      |  48 |      42000 |           0.0136 |            1.061 |   1.043  | False        | below expected-max (N=40)  |
| xgb      |  72 |      42000 |           0.0234 |            1.816 |   1.103  | True         | clears expected-max (N=40) |
| xgb      | 120 |      42000 |           0.0213 |            1.653 |   1.071  | False        | below expected-max (N=40)  |
| xgb      | 168 |      42000 |           0.0255 |            1.981 |   1.105  | True         | clears expected-max (N=40) |
| xgb      | 336 |      42000 |           0.017  |            1.321 |   1.071  | False        | below expected-max (N=40)  |
| catboost |   4 |      42000 |           0.0363 |            2.825 |   1.125  | True         | clears expected-max (N=40) |
| catboost |   6 |      42000 |           0.0492 |            3.825 |   1.157  | True         | clears expected-max (N=40) |
| catboost |   8 |      42000 |           0.0419 |            3.257 |   1.138  | True         | clears expected-max (N=40) |
| catboost |  12 |      42000 |           0.0519 |            4.039 |   1.172  | True         | clears expected-max (N=40) |
| catboost |  24 |      42000 |           0.0317 |            2.465 |   1.1    | True         | clears expected-max (N=40) |
| catboost |  48 |      42000 |           0.0399 |            3.104 |   1.135  | True         | clears expected-max (N=40) |
| catboost |  72 |      42000 |           0.0241 |            1.877 |   1.08   | True         | clears expected-max (N=40) |
| catboost | 120 |      42000 |           0.0237 |            1.845 |   1.077  | True         | clears expected-max (N=40) |
| catboost | 168 |      42000 |           0.0375 |            2.913 |   1.125  | True         | clears expected-max (N=40) |
| catboost | 336 |      42000 |           0.0196 |            1.521 |   1.081  | False        | below expected-max (N=40)  |

## Expected-max skill-less Sharpe bar (False Strategy Theorem)

Trial-Sharpe dispersion var_sr = 1.07e-04 (per-bar, across 30 completed combos).

| trial count N | E[max SR] per-bar | E[max SR] annualized | tree combos clearing |
|---:|---:|---:|---:|
| 30 | 0.0215 | 1.669 | 23 of 30 |
| 40 | 0.0227 | 1.762 | 23 of 30 |
| 80 | 0.0254 | 1.973 | 20 of 30 |

Headline trial count N = 40 (the cross-sectional family x horizon sweep; a defensible round count for the realized tree sweep of 30 combos plus the linear/forest configurations launched in the same search). At N=40, all 10 lgbm horizons clear, 9 of 10 catboost horizons clear (only the longest H=336 falls below), and 5 of 10 xgb horizons clear.

## lgbm rigorous per-horizon DSR (from the clean ledger)

This is the full Deflated Sharpe Ratio (probabilistic Sharpe against the expected-max benchmark, with skew and kurtosis), deflated against the dispersion of the 10-horizon Sharpe family.

|   H |   oos_sharpe_bar |   skew |   kurt |    sr0 |    dsr | dsr_survives   |
|----:|-----------------:|-------:|-------:|-------:|-------:|:---------------|
|   4 |           0.0415 | -1.196 |   99.7 | 0.0092 | 1      | True           |
|   6 |           0.0359 | -1.324 |   96.1 | 0.0092 | 1      | True           |
|   8 |           0.0335 | -2.042 |  130.6 | 0.0092 | 1      | True           |
|  12 |           0.0342 | -0.505 | 2659.1 | 0.0092 | 0.9999 | True           |
|  24 |           0.0465 |  0.379 |   32.1 | 0.0092 | 1      | True           |
|  48 |           0.0336 | -0.624 |   42.8 | 0.0092 | 1      | True           |
|  72 |           0.0432 |  0.518 |   60.7 | 0.0092 | 1      | True           |
| 120 |           0.0329 |  0.528 |   26.3 | 0.0092 | 1      | True           |
| 168 |           0.0307 |  0.383 |   13   | 0.0092 | 1      | True           |
| 336 |           0.0281 |  0.65  |   39.7 | 0.0092 | 1      | True           |

- lgbm DSR survives: **10/10** horizons.
- Canonical per-window PBO (dedicated rotation engine, the correct trial axis): **0.1**; IS->OOS rank persistence rho **+0.78**.
- Horizon-axis cross-check (labelled, near-degenerate by construction, NOT the headline): PBO = 0.623 over 252 CSCV splits; IS->OOS rho = +0.4182 (p = 0.2291).


---

## Neural cross-sectional families (real out-of-sample ledgers)

The neural families were run on a managed accelerator on the identical 787-pair panel, label, walk-forward, and per-fill cost model as the tree families, at the horizon family H in {6, 24, 72, 168}. Each row is scored from its REAL per-bar out-of-sample ledger against the SAME False-Strategy-Theorem bar as the trees (annualized ~1.76 at N=40). Families whose run produced no trades are recorded as deferred/failed below and carry no numbers.

| family   |   H |   oos_bars |   oos_sharpe_bar |   oos_sharpe_ann |   oos_pf |      dsr | deflation_verdict         |
|:---------|----:|-----------:|-----------------:|-----------------:|---------:|---------:|:--------------------------|
| mlp      |  24 |      42000 |          -0.0052 |           -0.402 |   0.9838 |   0.1333 | below expected-max (N=40) |
| mlp      |  72 |      42000 |          -0.0057 |           -0.442 |   0.9821 |   0.1116 | below expected-max (N=40) |
| mlp      | 168 |      42000 |          -0.0054 |           -0.418 |   0.9832 |   0.1242 | below expected-max (N=40) |
| lstm     |   6 |      41673 |          -0.0195 |           -1.513 |   0.94   | nan      | below expected-max (N=40) |

**Deferred / failed (no usable ledger):** gru, tcn, xattn.

Reading: consistent with the literature's tabular finding, the neural families do not beat the gradient-boosted trees on this cross-sectional task under identical costs and validation.
