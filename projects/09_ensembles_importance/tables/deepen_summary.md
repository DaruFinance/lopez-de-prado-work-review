# Project 09 — DEEPENING summary

Instruments: 40 ({'crypto': 27, 'forex': 8, 'equities': 5})


## (A) Bagging vs boosting (paired across instruments)

IS-OOS overfit gap, NLL-tuned: med(rf_gap_nll)=0.1139 med(hgb_gap_nll)=0.5389  rf_gap_nll<hgb_gap_nll in 100%  Wilcoxon p=1.82e-12
IS-OOS overfit gap, ACC-tuned: med(rf_gap_acc)=0.1320 med(hgb_gap_acc)=0.5832  rf_gap_acc<hgb_gap_acc in 100%  Wilcoxon p=1.82e-12
OOS DSR: med(rf_dsr)=0.0050 med(hgb_dsr)=0.0030  rf_dsr<hgb_dsr in 45%  Wilcoxon p=7.35e-01
annual SR: med(rf_sr_ann)=-0.6724 med(hgb_sr_ann)=-0.8159  rf_sr_ann<hgb_sr_ann in 45%  Wilcoxon p=6.85e-01

IS vs OOS log-loss LEVELS by market (overfit = OOS >> IS):
| market   |   rf_ll_is |   rf_ll_oos |   hgb_ll_is |   hgb_ll_oos |
|:---------|-----------:|------------:|------------:|-------------:|
| crypto   |     0.5686 |      0.68   |      0.2545 |       0.8085 |
| equities |     0.1338 |      0.5569 |      0.1532 |       0.6587 |
| forex    |     0.5692 |      0.6771 |      0.2773 |       0.808  |

Median HGB/RF overfit-gap ratio = 4.82x (boosting overfits ~4.8x more than bagging).
Fraction of instruments where RF gap < HGB gap: 100%.

DSR verdict: best per-instrument DSR over BOTH models = max median 0.025; instruments with EITHER model DSR>0.95: 0/40.

## (B) Feature importance: MDI vs MDA vs clustered-MDA

Substitution bias = Spearman rho(importance, feature mean |corr|). High rho = importance leaks to correlated features (LdP's MDI warning).
| market   |   mdi_bias |   mda_bias |   cmda_bias |
|:---------|-----------:|-----------:|------------:|
| crypto   |      0.491 |      0.115 |       0.552 |
| equities |      0.358 |      0.006 |       0.439 |
| forex    |      0.667 |      0.042 |       0.517 |

overall medians: MDI bias=0.485, MDA bias=0.073, clustered-MDA bias=0.517
MDA vs MDI substitution bias: med(mda_bias)=0.0727 med(mdi_bias)=0.4848  mda_bias<mdi_bias in 88%  Wilcoxon p=1.22e-06
clustered-MDA vs MDI substitution bias: med(cmda_bias)=0.5166 med(mdi_bias)=0.4848  cmda_bias<mdi_bias in 52%  Wilcoxon p=2.21e-01

### Stability of top-3 selected set across CPCV paths (mean Jaccard)

| market   |   stab_mdi |   stab_mda |   stab_cmda |   stab_random |
|:---------|-----------:|-----------:|------------:|--------------:|
| crypto   |      0.585 |      0.264 |       0.459 |         0.201 |
| equities |      0.438 |      0.275 |       0.367 |         0.201 |
| forex    |      0.566 |      0.252 |       0.505 |         0.201 |

overall medians: MDI=0.582, MDA=0.263, clustered-MDA=0.463, random baseline=0.201
MDA stability vs random selection: med(stab_mda)=0.2633 med(stab_random)=0.2005  stab_mda<stab_random in 2%  Wilcoxon p=5.21e-08
clustered-MDA vs MDA stability: med(stab_cmda)=0.4626 med(stab_mda)=0.2633  stab_cmda<stab_mda in 0%  Wilcoxon p=1.82e-12

MDA - random (paired): median +0.063, above random in 98% of instruments, Wilcoxon p=5.21e-08.
Median MDA stability z vs random = 3.49 sigma.

## (C) Tuning objective control: neg-log-loss vs accuracy

RF overfit gap: NLL-tuned vs ACC-tuned: med(rf_gap_nll)=0.1139 med(rf_gap_acc)=0.1320  rf_gap_nll<rf_gap_acc in 40%  Wilcoxon p=4.38e-04
HGB overfit gap: NLL-tuned vs ACC-tuned: med(hgb_gap_nll)=0.5389 med(hgb_gap_acc)=0.5832  hgb_gap_nll<hgb_gap_acc in 48%  Wilcoxon p=1.32e-04

NLL vs ACC selected the SAME RF config in 60% of instruments, same HGB config in 52%.
RF resulting SR: NLL-tuned vs ACC-tuned: med(rf_sr_ann)=-0.6724 med(rf_sr_ann_acc)=-0.5306  rf_sr_ann<rf_sr_ann_acc in 38%  Wilcoxon p=6.43e-04
