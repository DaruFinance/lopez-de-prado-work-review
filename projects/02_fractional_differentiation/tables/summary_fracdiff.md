# Fractional Differentiation, cross-sectional summary

- Pairs with a valid d* on the [0,1] grid: **505** (of 568 candidate 1h perps; rest too short or no d* found)

- **Median d\* = 0.150**  (Q1 0.100, Q3 0.150)

- Fraction of pairs with d\* < 1: **100.0%**

- Median memory retained (corr w/ level) at d\*: **0.981**

- Median |memory| at d=1 (plain returns): **0.010**  → returns destroy ~99% of the memory that FFD(d\*) keeps

- BTCUSDT: d\* = **0.15**, ADF = -3.66 (95% crit -2.86), corr w/ level = 0.987, window = 3901 bars

- d\* vs per-bar vol: Spearman -0.12;  d\* vs trend strength: Spearman +0.31


## tau sensitivity (BTC)

|    tau |   d_star |   window_len_at_dstar |   adf_at_dstar |   corr_level_at_dstar |
|-------:|---------:|----------------------:|---------------:|----------------------:|
| 0.001  |     0.25 |                    71 |       -3.09062 |              0.999364 |
| 0.0001 |     0.2  |                   497 |       -3.64403 |              0.997426 |
| 1e-05  |     0.15 |                  3901 |       -3.66443 |              0.987026 |


## d* quantiles across the cross-section

|       |      d_star |
|:------|------------:|
| count | 505         |
| mean  |   0.120099  |
| std   |   0.0613747 |
| min   |   0         |
| 5%    |   0         |
| 25%   |   0.1       |
| 50%   |   0.15      |
| 75%   |   0.15      |
| 95%   |   0.2       |
| max   |   0.25      |