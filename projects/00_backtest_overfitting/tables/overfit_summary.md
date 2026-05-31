# Backtest overfitting at scale — median by market

MA-crossover grid optimised per instrument (each combo = one trial), net of costs, on information-driven bars. best_sr_ann = best in-sample annualised Sharpe; sr0_ann = expected-max Sharpe of skill-less trials (False Strategy Theorem); DSR = deflated Sharpe (prob the winner is real); PBO = prob. of backtest overfitting; eff_trials = independent trials (ONC).

| market   |   instruments |   trials |   best_sr_ann |   sr0_ann |   dsr |   pbo |   eff_trials |
|:---------|--------------:|---------:|--------------:|----------:|------:|------:|-------------:|
| crypto   |             8 |      136 |         0.621 |     0.659 | 0.441 | 0.7   |            3 |
| equity   |             7 |      136 |         0.306 |     0.436 | 0.316 | 0.384 |            3 |
| forex    |             8 |      136 |         0.531 |     1.061 | 0.103 | 0.369 |            3 |