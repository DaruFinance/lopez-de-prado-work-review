# Bet Sizing (LdP AFML Ch.10) — results

_32 IS-tunable trials per instrument; DSR is the headline metric._


## By-market summary

| market   |   n_inst |   med_fixed_pf |   med_prob_pf |   med_disc_pf |   med_fixed_dsr |   med_prob_dsr |   med_disc_dsr |   n_prob_dsr_gt95 |   n_disc_dsr_gt95 |   med_fixed_turn |   med_prob_turn |   med_disc_turn |   med_pbo_prob |
|:---------|---------:|---------------:|--------------:|--------------:|----------------:|---------------:|---------------:|------------------:|------------------:|-----------------:|----------------:|----------------:|---------------:|
| crypto   |        1 |         0.4279 |        0.4279 |      nan      |          0.0691 |         0.0691 |       nan      |                 0 |                 0 |               24 |          1.0223 |             0   |              1 |
| equities |        1 |         0.5709 |        0.5709 |      nan      |          0.0708 |         0.0708 |       nan      |                 0 |                 0 |               40 |          2.1309 |             0   |              1 |
| forex    |        1 |         1.1632 |        0.8223 |        0.8481 |          0.7577 |         0.2658 |         0.2449 |                 0 |                 0 |              146 |          8.0716 |             7.4 |              1 |


## Per-instrument (head)

| market   | name    |   n_ev |   frac_act |   fixed_pf |   prob_pf |   disc_pf |   fixed_sr_ann |   prob_sr_ann |   disc_sr_ann |   fixed_dsr |   prob_dsr |   disc_dsr |   fixed_turn |   prob_turn |   disc_turn |   pbo_prob |   eff_n | best_prob                          |
|:---------|:--------|-------:|-----------:|-----------:|----------:|----------:|---------------:|--------------:|--------------:|------------:|-----------:|-----------:|-------------:|------------:|------------:|-----------:|--------:|:-----------------------------------|
| crypto   | BTCUSDT |     72 |     0.1667 |     0.4279 |    0.4279 |  nan      |        -2.0261 |       -2.0261 |        0      |      0.0691 |     0.0691 |   nan      |           24 |      1.0223 |         0   |          1 |       2 | f20/s60 pt1.0/sl1.0 h50 mt0.5 d0.1 |
| equities | SPY     |     61 |     0.3279 |     0.5709 |    0.5709 |  nan      |        -0.6454 |       -0.6454 |        0      |      0.0708 |     0.0708 |   nan      |           40 |      2.1309 |         0   |          1 |       2 | f20/s60 pt1.0/sl1.0 h50 mt0.5 d0.1 |
| forex    | EURUSD  |     73 |     1      |     1.1632 |    0.8223 |    0.8481 |         0.9173 |       -0.8089 |       -0.6485 |      0.7577 |     0.2658 |     0.2449 |          146 |      8.0716 |         7.4 |          1 |       2 | f20/s60 pt1.0/sl1.0 h50 mt0.5 d0.1 |