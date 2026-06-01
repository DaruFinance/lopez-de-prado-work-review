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

## Equity + Forex strategy corpora (for the at-scale overfitting harness)
`lib/ta_grid.py` is a Numba-accelerated TA-strategy-grid engine that sweeps the
market-agnostic families LdP-style rules use — SMA/EMA crossover, RSI level,
MACD signal-cross, ATR-channel breakout, Stochastic — over fast/slow/threshold +
stop-loss/take-profit grids (long-only and long-short). Each tuple is one
structural strategy: ~2,048 (default) / ~6,296 (`--wide`) per instrument.
- **Causal**: signal from bar *t*'s close fills at *t+1* open (no same-bar fill).
- **Intrabar OHLC exits**: SL/TP checked against bar high/low (never close-only),
  SL prioritised within a bar; gap-through at the open handled.
- **Costed**: 5 bp fee + 3 bp slip per fill (round trip = 2 fills); funding = 0.
- **Session-aware (equities)**: positions force-flat at the last bar of each RTH
  day; day ids on the NY date. Forex uses the UTC date (24h) and a tick clock.
- Hot loops (`_sim_grid_kernel` + the `_ema_k/_rsi_k/_atr_k/_stoch_k_k`
  indicators) are `@njit`; **verified bit-identical** vs a pure-NumPy reference
  (`_sim_one_ref` and frozen indicator refs): max|Δpnl|=0.0, max|Δn_trades|=0.
  Numba speedup ~21× steady-state (32s→1.5s/instrument; indicators were the hot
  path, not the sim).

Generate: `scripts/gen_equity_forex_corpora.py` (`--smoke` 1+1 tiny / `--profile`
cProfile / `--wide` heavy; idempotent per-asset `_DONE`). Heavy run queued via
`projects/00_backtest_overfitting/run_full_equity_forex_corpora.sh` (~107k
strategies = 9 equity ×6,296 + 8 forex ×6,296; ~10 min 1-core / ~3-4 min ×4;
~0.8 GB; one instrument in RAM at a time, ~1.5 GB peak). Output is the exact
harness layout `…/pnl_daily/asset=<TICKER>_equity|<PAIR>_fx/part-*.parquet`
(asset, family, strategy_name, date, pnl_sum, n_trades). `run_corpus_overfit.py`
`discover_crypto()` was hardened to exclude `*_equity`/`*_fx` so the crypto study
stays crypto-only.

## Layout
```
ldp_review/
  lib/        bars.py barstats.py fracdiff.py style.py fetch_1m.py fetch_fx_histdata.py
  data_cache/ run logs
  projects/NN_name/{scripts,figures,tables,writeup}
```
Crypto 1m cache: `…/Desktop/ldp_cache_1m/`. FX cache: `…/Desktop/ldp_cache_fx/`.
