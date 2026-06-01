# Project 7, DEEPEN headline

## (1) Family signal by market (mean OOS Sharpe, sign-test p)

| market   | family       |   mean_sharpe_clock |   median_sharpe_clock |   mean_sharpe_cusum |   n_inst |   frac_pos |   signtest_p |
|:---------|:-------------|--------------------:|----------------------:|--------------------:|---------:|-----------:|-------------:|
| crypto   | breaks(SADF) |             -0.055  |               -0.055  |             -0.076  |        2 |     0.5    |     nan      |
| crypto   | entropy      |             -0.0402 |               -0.0787 |             -0.0413 |        6 |     0.3333 |       0.6875 |
| equities | breaks(SADF) |              0.0021 |                0.0021 |              0.0036 |        2 |     0.5    |     nan      |
| equities | entropy      |              0.0114 |                0.0816 |             -0.0092 |        6 |     0.6667 |       0.6875 |
| forex    | breaks(SADF) |              0.0343 |                0.0343 |              0.0594 |        2 |     0.5    |     nan      |
| forex    | entropy      |             -0.0123 |               -0.0308 |             -0.0015 |        6 |     0.5    |       1      |

## (2) CUSUM-event sampling vs raw clock (mean OOS Sharpe)

| market   |   mean_clock |   mean_cusum |   mean_delta |   n |
|:---------|-------------:|-------------:|-------------:|----:|
| crypto   |      -0.0439 |      -0.05   |      -0.006  |   8 |
| equities |       0.0091 |      -0.006  |      -0.0151 |   8 |
| forex    |      -0.0007 |       0.0137 |       0.0144 |   8 |

## (3) Regime-rule DSR/PBO headline

- **best_rule**: R_expl_dir
- **best_instrument**: XLK
- **best_sharpe**: 0.2060766026547561
- **dsr**: 0.3104009111351925
- **sr0**: 0.22198517481397206
- **n_trials**: 18
- **pbo**: 0.1626984126984127

## (4) Conditional forward-return by regime (per market)

| market   | regime       |    n |   mean_fwd_ret |       se |      t_p |
|:---------|:-------------|-----:|---------------:|---------:|---------:|
| crypto   | explosive    |  129 |      -0.00129  | 0.001781 | 0.47011  |
| crypto   | nonexplosive | 1263 |      -0.000138 | 0.000436 | 0.752045 |
| crypto   | lo_entropy   |  555 |       0.000895 | 0.000684 | 0.191388 |
| crypto   | hi_entropy   |  837 |      -0.001001 | 0.000549 | 0.068487 |
| equities | explosive    |  234 |       0.007117 | 0.000883 | 0        |
| equities | nonexplosive | 1159 |      -0.003073 | 0.001354 | 0.023406 |
| equities | lo_entropy   |  554 |       0.002479 | 0.000577 | 2e-05    |
| equities | hi_entropy   |  839 |      -0.003897 | 0.001851 | 0.03552  |
| forex    | explosive    |  216 |       0.000355 | 0.000204 | 0.083294 |
| forex    | nonexplosive | 1177 |      -0.000135 | 8.4e-05  | 0.108401 |
| forex    | lo_entropy   |  585 |      -5.1e-05  | 0.00012  | 0.672418 |
| forex    | hi_entropy   |  808 |      -6.4e-05  | 0.000102 | 0.526689 |
