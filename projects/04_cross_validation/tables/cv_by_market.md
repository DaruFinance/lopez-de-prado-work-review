# Cross-validation leakage at scale, mean by market

RandomForest (200 trees, depth 5, leaf 50) on causal features, fixed-horizon labels with overlap span H=50 bars, 6-fold CV, embargo 1%. `kfold_*` = standard sklearn KFold (leaky); `purged_*` = purged k-fold + embargo (clean); `infl_*` = kfold − purged (the leakage). AUC is the sensitive detector; accuracy near a 50%% base rate is high-variance.

| market   |   instruments |   n_obs |   kfold_acc |   purged_acc |   infl_acc |   kfold_auc |   purged_auc |   infl_auc |
|:---------|--------------:|--------:|------------:|-------------:|-----------:|------------:|-------------:|-----------:|
| crypto   |             8 |    8429 |      0.5136 |       0.5152 |    -0.0017 |      0.5389 |       0.5365 |     0.0024 |
| equity   |             7 |   42117 |      0.5402 |       0.5392 |     0.001  |      0.5121 |       0.5144 |    -0.0024 |
| forex    |             8 |    8414 |      0.5168 |       0.5173 |    -0.0005 |      0.5282 |       0.5259 |     0.0023 |

## Inflation vs label overlap H, AUC inflation (pp), the clean signal

| market   |    1 |    5 |    10 |    25 |    50 |   100 |   150 |
|:---------|-----:|-----:|------:|------:|------:|------:|------:|
| crypto   | 0.03 | 0.24 |  0.19 |  0.21 |  0.53 |  0.36 |  1.69 |
| equity   | 0.02 | 0.04 | -0.01 | -0.09 | -0.47 | -0.46 | -0.03 |
| forex    | 0.12 | 0.17 | -0    |  0.27 |  0.45 |  1.07 |  0.85 |

## Inflation vs label overlap H, accuracy inflation (pp), noisy

| market   |     1 |     5 |    10 |    25 |   50 |   100 |   150 |
|:---------|------:|------:|------:|------:|-----:|------:|------:|
| crypto   | -0.24 | -0.01 |  0.2  |  0.12 | -0.6 | -0.01 |  0.33 |
| equity   |  0.07 |  0.04 | -0.01 |  0.04 |  0.1 | -0    |  0.12 |
| forex    |  0.11 |  0.07 | -0.23 | -0    | -0.1 | -0.07 |  0.3  |

## Residual inflation vs embargo, AUC (pp). Note: at fixed H=50 the overlap is
already fully purged, so added embargo mostly removes training data rather than leakage.

| market   |   0.0 |   0.005 |   0.01 |   0.02 |   0.05 |   0.1 |
|:---------|------:|--------:|-------:|-------:|-------:|------:|
| crypto   |  0.44 |    1.02 |   0.64 |   0.63 |   1.35 |  0.51 |
| equity   | -0.07 |   -0.06 |  -0.33 |  -0.29 |  -0.38 | -0.42 |
| forex    |  0.11 |    0.09 |   0.32 |   0.7  |   0.89 |  0.87 |