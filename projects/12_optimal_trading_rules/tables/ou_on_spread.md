# OU optimal trading rule on a cointegrated residual spread

_OU's true regime: the residual spread of a cointegrated crypto perp pair. Hedge ratio (beta) and OU parameters fit on TRAIN only; OOS trading with full intrabar OHLC exits (spread extremes bounded by leg OHLC) and per-leg costs on both legs. DSR is the headline._


## Per pair

| sym_a     | sym_b    | cointegrated   |   adf_p |   beta |   ou_half_life |   ou_phi |   n_oos_trades |   pt_star |   sl_star |   ou_sr_ann |   bb_sr_ann |   bh_sr_ann |   ou_pf |   bb_pf |   bh_pf |   ou_dsr |   bb_dsr |
|:----------|:---------|:---------------|--------:|-------:|---------------:|---------:|---------------:|----------:|----------:|------------:|------------:|------------:|--------:|--------:|--------:|---------:|---------:|
| 1INCHUSDT | UNIUSDT  | True           |  0      | 1.3356 |        95.8945 |   0.9928 |             86 |      0.25 |       0.5 |      9.2073 |      3.1092 |     -0.0255 |  2.1088 |  1.4345 |  0.9991 |   1      |   1      |
| THETAUSDT | FILUSDT  | True           |  0.0004 | 0.6911 |       116.868  |   0.9941 |             85 |      0.25 |       0.5 |     -4.6016 |     -3.5537 |     -0.5433 |  0.4964 |  0.5711 |  0.9827 |   0      |   0      |
| COMPUSDT  | AAVEUSDT | True           |  0.0014 | 1.2346 |       119.616  |   0.9942 |            108 |      0.25 |       0.5 |     10.7967 |      5.3144 |     -0.6817 |  2.7241 |  1.6432 |  0.9785 |   1      |   1      |
| SNXUSDT   | AAVEUSDT | True           |  0.0031 | 0.974  |        89.2645 |   0.9923 |             97 |      0.25 |       0.5 |     12.6631 |      6.7544 |     -1.1445 |  2.5236 |  1.6354 |  0.9625 |   1      |   1      |
| GALAUSDT  | SANDUSDT | True           |  0.0048 | 1.0549 |       174.178  |   0.996  |             91 |      0.25 |       0.5 |      4.5772 |     -2.1793 |     -0.3162 |  2.0907 |  0.76   |  0.9891 |   0.9255 |   0.0014 |


## Pooled

|                  |    value |
|:-----------------|---------:|
| n_pairs          |   5      |
| med_ou_sr_ann    |   9.2073 |
| med_bb_sr_ann    |   3.1092 |
| med_bh_sr_ann    |  -0.5433 |
| med_ou_pf        |   2.1088 |
| med_bb_pf        |   1.4345 |
| med_ou_dsr       |   1      |
| med_bb_dsr       |   1      |
| ou_beats_band_sr |   4      |
| ou_beats_bh_sr   |   4      |
| n_ou_dsr_gt95    |   3      |
| n_bb_dsr_gt95    |   3      |
| med_half_life    | 116.868  |