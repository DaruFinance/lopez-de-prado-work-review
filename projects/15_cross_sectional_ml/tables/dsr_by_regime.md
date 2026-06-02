# DSR by ML regime: single-series vs cross-sectional

| regime                         | family               |   n_strategies |   pbo |   effective_n | dsr_survives      |   is_oos_rank_rho |   median_oos_pf | verdict                                                       |
|:-------------------------------|:---------------------|---------------:|------:|--------------:|:------------------|------------------:|----------------:|:--------------------------------------------------------------|
| single-series (ML weakest)     | pooled (M1..M9 band) |          25162 |  0.62 |            40 | 0 of 25,162       |            nan    |        nan      | fails deflation                                               |
| cross-sectional (ML strongest) | lgbm                 |             10 |  0.1  |             3 | 10 of 10 horizons |              0.78 |          1.1234 | survives deflation; approaches but does not beat static ~1.17 |

_PBO via CSCV; effective-N via eigenvalue participation ratio; DSR via the False Strategy Theorem benchmark, all from the program's deflation toolkit._

**Notes.** Headline PBO and IS->OOS rank-persistence are the canonical per-window figures from the dedicated engines (the correct trial axis). `effective_n` is measured on different axes (single-series: ~40 independent of 25,162 strategies; cross-sectional: independent horizon-bets of 10), so the two are not directly comparable. The best static structural carry/momentum benchmark sits at ~PF 1.17; cross-sectional ML approaches but does not beat it.

**Multi-family corroboration.** The cross-sectional survival is not a single-model artefact. Two further gradient-boosted-tree families were run end-to-end on the same 787-pair survivorship-honest panel and the same purged walk-forward: under the False Strategy Theorem deflation bar (expected-max skill-less Sharpe over the family x horizon sweep, N=40), lgbm clears 10 of 10 horizons, catboost clears 9 of 10, and xgb clears 5 of 10 - 23 of 30 tree (family, horizon) combos in total. Single-series ML clears 0 of ~25,162. The per-family table is in family_dsr.md; the deflation bar and the per-horizon DSR detail are computed in lib/overfit.py. The deferred families (random forest, extra-trees, the linear set, and the neural set) did not complete on the full panel and carry no result; see the writeup for the reasons.
