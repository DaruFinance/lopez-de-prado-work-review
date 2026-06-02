# Full assembly-line pipeline on held-out time

_One disclosed system: dollar bars -> triple-barrier labels -> uniqueness weights -> bagged-tree meta-model -> meta-label gate -> bet sizing -> Hierarchical Risk Parity -> Deflated-Sharpe deploy gate. The final out-of-time window is held out and scored once._


## Headline (pooled out-of-time window)

| process | OOS Sharpe (ann) | OOS PF | OOS DSR | deploys? |
|---|---|---|---|---|
| assembled pipeline (DSR-gated) | 0.000 | n/a | 0.000 | NO |
| Sisyphus best-in-sample pick | -0.827 | 0.959 | (no deflation) | deploys blindly |
| buy and hold | 1.810 | 1.058 | - | - |

Program disclosed 162 configurations searched across 6 instruments. Sleeves deployed by the gate: 0 (none).


## Per-instrument station pass-through (selected configuration)

| name     |   n_bars |   n_ev |   n_ev_tune |   n_ev_oos |   base_rate |   avg_uniqueness |   eff_sample |   naive_n |   frac_act_oos |   tune_sr_ann |   tune_dsr | deploy   |   pipe_oos_sr_ann |   sis_oos_sr_ann |   bh_oos_sr_ann | best_cfg                |
|:---------|---------:|-------:|------------:|-----------:|------------:|-----------------:|-------------:|----------:|---------------:|--------------:|-----------:|:---------|------------------:|-----------------:|----------------:|:------------------------|
| BNBUSDT  |    19984 |    217 |         170 |         47 |      0.53   |           0.9745 |      211.464 |       217 |         0.617  |       -0.5897 |     0.0001 | False    |           -0.4063 |          -0.8394 |          0.364  | f30/s90 pt2.0/sl1.5 h25 |
| BTCUSDT  |    19995 |    715 |         587 |        128 |      0.5538 |           0.9375 |      670.327 |       715 |         0.8047 |        0.8988 |     0.0708 | False    |           -1.5918 |          -0.7118 |          1.2791 | f10/s30 pt2.0/sl1.5 h25 |
| ETHUSDT  |    19998 |    379 |         300 |         79 |      0.5251 |           0.9714 |      368.146 |       379 |         0.6962 |        0.1832 |     0.0004 | False    |           -3.0354 |          -2.4229 |          0.424  | f20/s60 pt2.0/sl1.5 h50 |
| LINKUSDT |    19996 |    693 |         546 |        147 |      0.544  |           0.9556 |      662.227 |       693 |         0.6667 |        0.0698 |     0.0276 | False    |            0.6629 |          -0.0308 |          0.5563 | f10/s30 pt2.0/sl1.5 h25 |
| SOLUSDT  |    19997 |    239 |         190 |         49 |      0.5021 |           0.9814 |      234.558 |       239 |         0.5306 |        1.8229 |     0.5704 | False    |           -3.2994 |          -1.24   |          1.0375 | f30/s90 pt2.0/sl1.5 h25 |
| XRPUSDT  |    19849 |    751 |         599 |        152 |      0.5606 |           0.9411 |      706.803 |       751 |         0.6974 |        2.0793 |     0.4626 | False    |            1.8639 |           0.5837 |          1.0831 | f10/s30 pt2.0/sl1.5 h25 |


_Stations: n_ev = triple-barrier labeled events; avg_uniqueness and eff_sample (= sum of uniqueness weights, versus naive_n events) are the sample-uniqueness station; frac_act_oos is the meta-label gate's act rate out of sample; tune_dsr is the per-instrument deploy gate; the three OOS Sharpe columns are the realized held-out result of the pipeline sleeve, the Sisyphus primary-only sleeve, and buy-and-hold._
