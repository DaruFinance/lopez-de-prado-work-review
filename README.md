# Author Work Review — Marcos López de Prado, reproduced and extended at scale

A research program that takes every method LdP holds a **strong opinion** about (see the source
catalog: `lopezdeprado_strong_opinions.md`) and, for each, **reproduces** it on real data,
**extends / stress-tests** it, and adds independent notes and opinions. Each project is
**multi-market** by rule — Crypto + US Equities and/or Forex, **never crypto-only**. Strong
projects graduate into website "project" pages and, where results warrant, standalone papers.

## Standing conventions
- **Real data only**, finest sensible granularity; causal transforms only (no lookahead).
- **Multi-market**: every project spans ≥2 asset classes.
- **Costed, walk-forward** evaluation for any strategy-level claim (crypto 5bp taker / 2bp maker /
  2bp slip + funding; per-market spreads for FX/equities). Statistical-property studies (bars,
  frac-diff) are cost-free by nature and labelled as such.
- **Honest findings**: confounds are isolated and negative results reported, not buried.
- Clean reusable code in `lib/`; per-project `scripts/ figures/ tables/ writeup/`.
- **Performance:** every heavy run is profiled (cProfile smoke test) before optimising; hot loops are
  moved to **Numba** njit kernels and verified bit-identical (timestamps exact; sums within float
  rounding) against the pandas reference. Bar aggregation (`bars._agg_kernel`) is the shared hot path
  — 139× over pandas.
- Research writeups keep real paths; **public website versions are scrubbed** (no internal paths,
  no proprietary-diagnostic or AI-assistance mentions) per house rules.

## Data assets
| Market | Source | Granularity | Coverage |
|---|---|---|---|
| Crypto | Binance USD-M perp dumps | 1m (2022-24) + 1h | 27 pairs clean 1m; 568 pairs 1h |
| US Equities | Algoseek ETF 1-min; Lean single-stock | 1m (to 2007) | SPY/QQQ/IWM + sector SPDRs + vol ETFs; single-stock minute/tick/daily w/ factors+shortable |
| Forex | HistData tick (quote ticks) | tick → 1m | 8 majors (2022-24); tick-count = information clock (spot FX has no volume) |

## Projects
| # | Project | LdP source | Markets | Status | Headline |
|---|---|---|---|---|---|
| 00 | Backtest overfitting & DSR (validation harness) | AFML Ch.7,11-15; DSR/PBO/False-Strategy papers | Crypto + Equities + Forex | **done** | best backtest Sharpe ≤ luck (False Strategy Thm) in all 3 markets; DSR 0.10-0.44 (none significant); PBO up to 0.70; 136 trials ≈ 3 independent bets. Harness verified vs LdP's math. |
| 01 | Information-driven bars | AFML Ch.2 | Crypto + Equities + Forex | **done** | exkurt: crypto 5.3→1.5, equities(RTH) 9.95→3.4, forex 2.98→1.06. Confirmed all 3 markets; equities need session-aware handling; FX uses tick clock; advantage is granularity-dependent; reproduction fragile to data quality |
| 02 | Fractional differentiation | AFML Ch.5 | Crypto + Equities + Forex | **done** | FFD keeps ~0.98 corr w/ level vs 0.01 for returns (universal); d\*≈0.10-0.15 at 1h — below LdP's 0.3-0.6 (frequency/length driven) |
| 03+ | Labeling & meta-labeling; Purged CV/CPCV; feature importance; bet sizing; OU rules; HRP/NCO; causal | AFML Ch.3-10,16; ML4AM; Causal Factor Investing | TBD | planned | per `lopezdeprado_application_plan.md` |

## Layout
```
ldp_review/
  lib/        bars.py barstats.py fracdiff.py style.py fetch_1m.py fetch_fx_histdata.py
  data_cache/ run logs
  projects/NN_name/{scripts,figures,tables,writeup}
```
Crypto 1m cache: `…/Desktop/ldp_cache_1m/`. FX cache: `…/Desktop/ldp_cache_fx/`.
