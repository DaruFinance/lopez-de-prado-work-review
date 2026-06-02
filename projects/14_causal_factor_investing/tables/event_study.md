# Event study -- perpetual-listing natural experiment

- Treated events (Binance USD-M perpetual listings, pre-spot vs post-perp): **170**
- Control majors (placebo, no listing event): BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, LTCUSDT
- Pre window: token spot daily returns (up to 60d before listing); Post window: perp daily returns (up to 60d after).
- Factor: short-term reversal (trailing 5d, lagged), identical to the main study.

## Regression-discontinuity in the reversal-factor effect

| group | window | reversal slope | t-stat | n |
|---|---|---|---|---|
| treated | pre | -0.0008 | -0.50 | 8671 |
| treated | post | -0.0050 | -6.49 | 9350 |
| treated | post-minus-pre | -0.0042 | -2.55 | 18021 |
| control | pre | -0.0025 | -3.02 | 8738 |
| control | post | -0.0019 | -3.28 | 9350 |
| control | post-minus-pre | 0.0006 | 0.58 | 18088 |

## Abnormal return (continuation sleeve)

- The abnormal-return sleeve bets on continuation (sign(factor) * return); a NEGATIVE value means recent winners reverse, i.e. reversal pays. Cumulative abnormal return over 30 post-listing days, net of each asset's own pre-listing baseline: **-14.90%** (bootstrap two-sided p = 0.021, 170 events). A negative CAR is the reversal effect strengthening, consistent with the more-negative post slope above.
- Per-event check (caveat): the reversal |t| is larger POST than PRE in only **48%** of individual events (82/170). The discontinuity is a pooled-panel / average phenomenon, not present asset-by-asset.

## Reading

- Treated discontinuity (post-minus-pre interaction) t = -2.55; control discontinuity t = 0.58.
- The reversal effect shifts discontinuously at the exogenous listing event in the treated assets but NOT in the controls -- consistent with the event (not the calendar) driving the change.
