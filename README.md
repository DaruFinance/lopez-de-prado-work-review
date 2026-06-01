# López de Prado, reproduced and extended at scale

This repository reproduces the quantitative-finance methods that Marcos López de
Prado argues for in *Advances in Financial Machine Learning* (AFML), *Machine
Learning for Asset Managers* (ML4AM), *Causal Factor Investing*, and the
associated papers, and then stress-tests them on real data across three asset
classes (crypto, US equities, forex) under a single hard evaluation standard.

Each method gets its own study folder. We implement the technique faithfully,
run it multi-market on real 1-minute (and daily) data, cost every strategy-level
claim, control for leakage with purged cross-validation, and judge the result by
the **Deflated Sharpe Ratio (DSR)** rather than raw Sharpe.

## The through-line

Across these methods, reproduced multi-market at scale with realistic costs and
DSR gating, almost nothing clears deflated significance anywhere. That is
precisely López de Prado's thesis: once you correct for the number and
correlation of the trials behind a result, the apparent edge usually evaporates.
What *does* reproduce cleanly are his methodological claims: the False Strategy
Theorem (the best backtested Sharpe is at or below what luck alone produces), the
Gaussianizing effect of information-driven bars, the memory/stationarity
trade-off of fractional differentiation, the leakage of naive k-fold CV, the bias
of MDI feature importance, and the direction-flipping of confounded factor
regressions. The value is in the process and the negative results, reported
plainly.

## Standing conventions

- **Real data only**, finest sensible granularity; causal transforms only (no
  look-ahead). No synthetic price series.
- **Multi-market by rule**: every study spans at least two asset classes,
  typically all three.
- **Costed, walk-forward** evaluation for any strategy-level claim (crypto 5 bp
  taker / 2 bp maker / 2 bp slip plus funding; per-market modeled spreads and
  commissions for FX and equities). Pure statistical-property studies (bars,
  fractional differentiation) are cost-free by nature and labelled as such.
- **Honest findings**: confounds are isolated and negative results are reported,
  not buried.
- **Leakage control**: purged k-fold / CPCV with embargo wherever labels overlap.
- **Performance**: heavy hot loops are moved to Numba `njit` kernels and verified
  bit-identical against a pure-NumPy/Python reference before use.
- Reusable code lives in `lib/`; each study has its own
  `scripts/ figures/ tables/ writeup/`.

## Finished studies

These nine studies are complete. The study numbers below are the reader-facing
numbers; each maps to a folder under `projects/` whose name follows an earlier
folder ordering (the mapping is given in the [Layout](#layout) table). Every
result table and figure in the repo was produced from the scripts in that folder.

**01 - Backtest Overfitting and the Deflated Sharpe Ratio.**
Across about 92,500 real strategies in crypto, equities and forex, the best beats
its multiple-testing null in none.
[`projects/00_backtest_overfitting/`](projects/00_backtest_overfitting/writeup/README.md)

**02 - Information-Driven Bars.**
Dollar, volume and tick bars Gaussianize returns in all three markets; the effect
is granularity-dependent, and equities need session handling.
[`projects/01_information_driven_bars/`](projects/01_information_driven_bars/writeup/README.md)

**03 - Fractional Differentiation.**
Fixed-width fractional differencing keeps about 0.98 correlation with the price
level versus about 0.01 for plain returns.
[`projects/02_fractional_differentiation/`](projects/02_fractional_differentiation/writeup/README.md)

**04 - Labeling and Cross-Validation.**
K-fold leakage scales with the ratio of label horizon to fold size; meta-labeling
is a precision filter, not alpha.
[`projects/03_meta_labeling/`](projects/03_meta_labeling/writeup/README.md),
[`projects/04_cross_validation/`](projects/04_cross_validation/writeup/README.md),
[`projects/05_trend_scanning/`](projects/05_trend_scanning/writeup/README.md)

**05 - Predictive Features.**
Structural-break, entropy and microstructure features carry weak signal that does
not survive cost plus deflation; a cheap proxy matches expensive order-flow data.
[`projects/07_structural_breaks_entropy/`](projects/07_structural_breaks_entropy/writeup/README.md),
[`projects/08_microstructural_features/`](projects/08_microstructural_features/writeup/README.md)

**06 - Ensembles and Feature Importance.**
Bagging generalizes about 4.8x better than boosting on 100% of 40 instruments;
MDI is substitution-biased, MDA is not.
[`projects/09_ensembles_importance/`](projects/09_ensembles_importance/writeup/README.md)

**07 - Trading Rules and Bet Sizing.**
Bet sizing cuts turnover 80 to 87% but adds no deflated edge; Triple-Penance
AR(1) drawdown control is the validated win.
[`projects/11_bet_sizing/`](projects/11_bet_sizing/writeup/README.md),
[`projects/12_optimal_trading_rules/`](projects/12_optimal_trading_rules/writeup/README.md)

**08 - Portfolio Construction: HRP, NCO and Denoising.**
HRP and NCO beat raw Markowitz on out-of-sample variance; the value of denoising
is a function of q = T/N.
[`projects/13_portfolio_construction/`](projects/13_portfolio_construction/writeup/README.md)

**09 - Causal Factor Investing.**
A confounder makes a null factor look significant 100% of the time; backdoor
adjustment fixes it, and few real factors survive.
[`projects/14_causal_factor_investing/`](projects/14_causal_factor_investing/writeup/README.md)

## Coming soon

These two studies are planned and not yet released.

- **Sample Uniqueness and Sequential Bootstrap.** Overlapping triple-barrier
  labels make observations non-IID; uniqueness weighting and the sequential
  bootstrap restore the effective sample size before training.
- **Meta-Strategy Organization.** The assembly-line model: specialized, separable
  research roles plus mandatory disclosure of every trial, as the structural
  antidote to the lone-quant backtest search.

## Data

No market data is bundled in this repository. All studies read from local data
roots that you supply; see **[`DATA.md`](DATA.md)** for the sources, schemas, and
coverage (Binance USD-M perpetual 1m/1h dumps, Algoseek ETF 1-minute trade bars,
HistData forex quote ticks resampled to a 1-minute base). Every result table and
figure already in the repo was produced from these real sources.

The at-scale strategy corpora behind study 01 (tens of thousands of costed,
per-strategy daily PnL series per market) are produced by a separate data
pipeline that is not part of this repository. The study scripts here read the
resulting per-strategy daily PnL from the `LDP_PNL_DAILY` root and perform the
overfitting, DSR, PBO and effective-trials analysis on it. The per-instrument
overfitting figures (study 01) regenerate end-to-end from the raw price roots.

## Setup and reproduce

Requirements: Python 3.10+, plus `numpy pandas scipy scikit-learn matplotlib
pyarrow numba`.

All data paths are centralized in **[`config.py`](config.py)**. It reads each
root from an environment variable and falls back to a repo-relative default under
`data/`. Point the variables at your own copies of the datasets:

```bash
export LDP_CRYPTO_1M=/path/to/binance_perp_1m       # *_1m.parquet
export LDP_CRYPTO_1H=/path/to/binance_perp_1h       # binance_um/*_1h.parquet
export LDP_CRYPTO_30M=/path/to/binance_perp_30m     # *_30m.parquet
export LDP_EQUITY_1M=/path/to/algoseek_etf_1min     # *.csv.gz
export LDP_FX_1M=/path/to/histdata_fx_1m            # *_fx1m.parquet
export LDP_FX_RAW=/path/to/fx_raw_csv               # per-pair raw FX CSVs
export LDP_PNL_DAILY=/path/to/pnl_daily             # per-strategy daily PnL corpus
```

Then run any study from the repository root, for example:

```bash
python3 projects/01_information_driven_bars/scripts/run_bars_multimarket.py
bash    projects/12_optimal_trading_rules/run_full.sh
```

Each study folder has a `run_full.sh` (where applicable) and a `writeup/README.md`
with the exact reproduce commands, the method, and the honest verdict.

## Layout

```
config.py            # central data-root configuration (env vars + relative defaults)
DATA.md              # data sources, schemas, coverage
lib/                 # shared, reusable code (bars, fracdiff, overfit/DSR, realism, ...)
projects/NN_name/
  scripts/           # study scripts
  figures/           # generated figures
  tables/            # generated result tables (CSV/MD)
  writeup/README.md  # method + results + verdict
```

Reader-facing study number to folder mapping:

| Study | Folder(s) under `projects/` |
|---|---|
| 01 Backtest Overfitting and DSR | `00_backtest_overfitting` |
| 02 Information-Driven Bars | `01_information_driven_bars` |
| 03 Fractional Differentiation | `02_fractional_differentiation` |
| 04 Labeling and Cross-Validation | `03_meta_labeling`, `04_cross_validation`, `05_trend_scanning` |
| 05 Predictive Features | `07_structural_breaks_entropy`, `08_microstructural_features` |
| 06 Ensembles and Feature Importance | `09_ensembles_importance` |
| 07 Trading Rules and Bet Sizing | `11_bet_sizing`, `12_optimal_trading_rules` |
| 08 Portfolio Construction | `13_portfolio_construction` |
| 09 Causal Factor Investing | `14_causal_factor_investing` |
