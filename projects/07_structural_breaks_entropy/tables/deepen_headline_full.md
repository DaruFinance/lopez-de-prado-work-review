# Project 7 — DEEPEN headline

## (1) Family signal by market (mean OOS Sharpe, sign-test p)

| market   | family       |   mean_sharpe_clock |   median_sharpe_clock |   mean_sharpe_cusum |   n_inst |   frac_pos |   signtest_p |
|:---------|:-------------|--------------------:|----------------------:|--------------------:|---------:|-----------:|-------------:|
| crypto   | breaks(SADF) |              0.0119 |                0.0206 |              0.0062 |       12 |     0.75   |       0.146  |
| crypto   | entropy      |             -0.0083 |               -0.0133 |             -0.0062 |       36 |     0.4444 |       0.6177 |
| equities | breaks(SADF) |             -0.0457 |               -0.0425 |             -0.056  |        7 |     0      |       0.0156 |
| equities | entropy      |             -0.033  |               -0.0254 |             -0.034  |       21 |     0.1905 |       0.0072 |
| forex    | breaks(SADF) |              0.0004 |                0.0033 |             -0.001  |        8 |     0.5    |       1      |
| forex    | entropy      |              0.0054 |                0.0206 |              0.0019 |       24 |     0.6667 |       0.1516 |

## (2) CUSUM-event sampling vs raw clock (mean OOS Sharpe)

| market   |   mean_clock |   mean_cusum |   mean_delta |   n |
|:---------|-------------:|-------------:|-------------:|----:|
| crypto   |      -0.0032 |      -0.0031 |       0.0001 |  48 |
| equities |      -0.0362 |      -0.0395 |      -0.0033 |  28 |
| forex    |       0.0041 |       0.0012 |      -0.003  |  32 |

## (3) Regime-rule DSR/PBO headline

- **best_rule**: R_loent_dir
- **best_instrument**: USDJPY
- **best_sharpe**: 0.0872762717495124
- **dsr**: 0.45602048205833123
- **sr0**: 0.08874372174969611
- **n_trials**: 81
- **pbo**: 0.2698412698412698

## (4) Conditional forward-return by regime (per market)

| market   | regime       |     n |   mean_fwd_ret |       se |      t_p |
|:---------|:-------------|------:|---------------:|---------:|---------:|
| crypto   | explosive    |  9850 |       0.002142 | 0.000541 | 7.5e-05  |
| crypto   | nonexplosive | 60903 |      -0.000378 | 0.000163 | 0.020872 |
| crypto   | lo_entropy   | 31745 |       0.00011  | 0.00025  | 0.659228 |
| crypto   | hi_entropy   | 39008 |      -0.000138 | 0.000206 | 0.502663 |
| equities | explosive    |  4946 |       0.002919 | 0.000496 | 0        |
| equities | nonexplosive | 36329 |       0.000972 | 0.000157 | 0        |
| equities | lo_entropy   | 18642 |       0.000827 | 0.000218 | 0.000153 |
| equities | hi_entropy   | 22633 |       0.001517 | 0.000207 | 0        |
| forex    | explosive    |  5475 |      -3.5e-05  | 6.6e-05  | 0.596177 |
| forex    | nonexplosive | 41696 |      -3e-06    | 2.2e-05  | 0.89178  |
| forex    | lo_entropy   | 21305 |       1.1e-05  | 3.2e-05  | 0.740803 |
| forex    | hi_entropy   | 25866 |      -2.1e-05  | 2.9e-05  | 0.461218 |
