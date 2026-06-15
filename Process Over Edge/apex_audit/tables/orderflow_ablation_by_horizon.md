# Order-flow ablation — full 10-horizon vector (clean-WFO baseline)

Saved artifact behind the apex ablation (the vector the paper Remark said was
not banked). Baseline = clean WFO with both defects fixed and all features;
ablated = identical pipeline with `of_buy` + `of_delta` (the 4h/8h forward-filled
order-flow block) dropped. Removing that one block collapses the profit factor at
every horizon — the economic magnitude was the leak.

| H (bars) | PF (with order-flow) | PF (ablated) | PF drop |
|---:|---:|---:|---:|
| 4 | 3.058 | 1.028 | 2.030 |
| 6 | 2.772 | 1.056 | 1.715 |
| 8 | 2.247 | 1.077 | 1.171 |
| 12 | 1.992 | 1.080 | 0.912 |
| 24 | 1.622 | 1.077 | 0.546 |
| 48 | 1.432 | 1.055 | 0.377 |
| 72 | 1.234 | 1.079 | 0.155 |
| 120 | 1.291 | 1.059 | 0.232 |
| 168 | 1.255 | 1.054 | 0.202 |
| 336 | 1.134 | 1.039 | 0.095 |
