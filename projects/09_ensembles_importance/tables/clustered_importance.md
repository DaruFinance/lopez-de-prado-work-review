# Clustered feature importance (crypto BTCUSDT)

Hierarchical (Ward) clustering on the 1-|corr| feature distance, importance measured per cluster. Same triple-barrier labeled task, RF recipe and purged CV as the main study.

RF config (neg-log-loss tuned): `{'max_features': 2, 'min_weight_fraction_leaf': 0.05, 'n_estimators': 200}`. 351 events, 10 features, base rate 0.536.

## Per cluster

| cluster   | members                |   n_members |   clustered_MDI |   clustered_MDA |   flat_MDI_sum |   flat_MDI_max_member |   flat_MDA_max_member |
|:----------|:-----------------------|------------:|----------------:|----------------:|---------------:|----------------------:|----------------------:|
| C1        | side, mom6, mom12, rsi |           4 |          0.3851 |          0.0735 |         0.3851 |                0.1484 |                0.0239 |
| C5        | vol, range_atr         |           2 |          0.2126 |          0.0317 |         0.2126 |                0.1133 |                0.0065 |
| C2        | ma_gap                 |           1 |          0.1529 |          0.0241 |         0.1529 |                0.1529 |                0.0154 |
| C4        | ofi                    |           1 |          0.0864 |          0.0092 |         0.0864 |                0.0864 |                0.0111 |
| C3        | mom3                   |           1 |          0.0943 |          0.0042 |         0.0943 |                0.0943 |                0.0002 |
| C6        | vol_ratio              |           1 |          0.0686 |         -0.0004 |         0.0686 |                0.0686 |               -0.002  |

## Per feature (flat numbers, for reference)

| feature   | cluster   |   flat_MDI |   flat_MDA |
|:----------|:----------|-----------:|-----------:|
| rsi       | C1        |     0.1484 |     0.0239 |
| mom12     | C1        |     0.1191 |     0.0047 |
| mom6      | C1        |     0.1055 |     0.0198 |
| side      | C1        |     0.0121 |     0.0009 |
| ma_gap    | C2        |     0.1529 |     0.0154 |
| mom3      | C3        |     0.0943 |     0.0002 |
| ofi       | C4        |     0.0864 |     0.0111 |
| vol       | C5        |     0.1133 |     0.0065 |
| range_atr | C5        |     0.0994 |     0.0026 |
| vol_ratio | C6        |     0.0686 |    -0.002  |

## Rank stability across purged folds (mean pairwise Spearman; higher = steadier selection)

| method        |   rank_stability_spearman |
|:--------------|--------------------------:|
| flat_MDI      |                    0.8917 |
| flat_MDA      |                    0.1208 |
| clustered_MDI |                    0.9695 |
| clustered_MDA |                    0.3029 |

## Concentration (de-dilution)

- Gini of flat per-feature MDI: **0.205**
- Gini of clustered MDI: **0.337**
- Largest correlated block C1 (4 features: side, mom6, mom12, rsi): best single member's flat MDI is **0.148**, but the block scored as one cluster is **0.385**.
