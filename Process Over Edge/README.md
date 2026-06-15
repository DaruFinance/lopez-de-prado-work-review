# Process over edge: López de Prado's methods, reproduced and stress-tested at scale

This repository reproduces the quantitative-finance methods of Marcos López de Prado,
from *Advances in Financial Machine Learning*, *Machine Learning for Asset Managers*,
*Causal Factor Investing*, and the associated papers, then tests them at full scale on
real data across three asset classes (crypto perpetual futures, US equities and ETFs, and
FX) under one hard evaluation standard. Every study has its own folder with the scripts
that produce its result.

The standard is uniform. Real data only, at the finest sensible granularity, with causal
transforms and no look-ahead. Every strategy-level claim is charged realistic costs and
evaluated strictly walk-forward with purged cross-validation. Nothing is called
significant on raw Sharpe; the gate is the Deflated Sharpe Ratio against the number and
correlation of trials actually searched.

## What the results say

The methods split by what each one is for. The tools for measuring and for not fooling
yourself reproduce cleanly: deflated significance and the False Strategy Theorem, purged
cross-validation, information-driven bars, fractional differentiation, sample-uniqueness
counting, bagging discipline, the backdoor adjustment for confounded factors, and
denoised allocation. The methods sold as sources of return do not survive cost and
deflation: meta-labeling, trend-scanning, structural-break and microstructure direction
signals, bet sizing, and the single-series optimal trading rule.

One result looked like an exception. A cross-sectional machine-learning rotation cleared
the deflated-Sharpe bar at all ten forecast horizons and was the program's only apparent
survivor. A from-scratch audit traced its entire economic magnitude to a data-handling
leak: a coarse-frequency order-flow feature forward-filled onto a finer grid, which a
one-bar lag does not decontaminate. Deflation cannot see that, because it asks whether a
number is too large for the size of a search, not whether the number is real. The audit is
in [`apex_audit/`](apex_audit/).

Two checks bound the conclusion. A synthetic [`positive_control/`](positive_control/)
confirms the acceptance gates discriminate, so they pass real planted edges and reject
noise rather than rejecting everything. And eighteen of our own practitioner extensions
went through the same pre-registered bar in [`extensions/`](extensions/); none manufactured
an edge.

## Standing conventions

- Real data only, finest sensible granularity, causal transforms, no synthetic price
  series in the results.
- Multi-market by rule: every study spans at least two asset classes, usually all three.
- Costed and walk-forward for any strategy-level claim (crypto 5 bp taker / 2 bp maker /
  2 bp slippage per fill plus funding; per-market modeled spreads, commissions, and borrow
  for equities and FX). Pure distributional studies (bars, fractional differentiation) are
  cost-free by nature and labelled as such.
- Leakage control: purged k-fold and combinatorial purged cross-validation with embargo
  wherever labels overlap, and pollute-and-verify on the procedure, not just the features.
- Multiple-testing control at the assembly step: Deflated Sharpe against an effective
  trial count, not the nominal one.
- Reusable code lives in `lib/`; each study folder holds its own scripts and a README.

## Studies

Each folder has a README with the method, the exact reproduce commands, and the verdict.

| Folder | Method | Verdict |
|---|---|---|
| [`projects/00_backtest_overfitting`](projects/00_backtest_overfitting) | Deflated Sharpe, False Strategy Theorem, PBO, effective-N | Reproduced; the keystone gate. Best Sharpe below its luck null in all three markets |
| [`projects/01_information_driven_bars`](projects/01_information_driven_bars) | Information-driven bars | Reproduced; excess kurtosis cut about 3 to 5x |
| [`projects/02_fractional_differentiation`](projects/02_fractional_differentiation) | Fractional differentiation | Reproduced; about 0.98 memory at stationarity versus 0.01 for returns |
| [`projects/03_meta_labeling`](projects/03_meta_labeling) | Triple-barrier and meta-labeling | Failed as alpha; lifts PF in 38/42, 0/42 clear DSR. A precision tool |
| [`projects/03b_metalabel_real_primary`](projects/03b_metalabel_real_primary) | Meta-label on an edged primary | Amplifies a real edge; DSR 0.78, short of 0.95 on a two-pair sample |
| [`projects/04_cross_validation`](projects/04_cross_validation) | Purged k-fold, embargo, CPCV | Reproduced; CPCV path spread about 50x the leak bias |
| [`projects/05_trend_scanning`](projects/05_trend_scanning) | Trend-scanning labels | Failed; PF win 28/42, deflation-neutral. The edge is horizon length |
| [`projects/07_structural_breaks_entropy`](projects/07_structural_breaks_entropy) | Structural breaks and entropy | Failed; DSR about 0.002, AUC about a coin flip, net-negative 22/24 |
| [`projects/08_microstructural_features`](projects/08_microstructural_features) | Microstructural features | Failed on direction; one hygiene win, a free BVC proxy matches paid order-flow |
| [`projects/09_ensembles_importance`](projects/09_ensembles_importance) | Bagging vs boosting, MDI vs MDA | Reproduced; bagging gap 4.8x tighter on 100% of 40 instruments |
| [`projects/11_bet_sizing`](projects/11_bet_sizing) | Bet sizing from probabilities | Failed as alpha; turnover down 80 to 87%, 0/42 clear DSR. A cost tool |
| [`projects/12_optimal_trading_rules`](projects/12_optimal_trading_rules) | OU trading rule and Triple-Penance | OU about a coin flip; Triple-Penance reproduced (IID MaxDD understated about 3.2x) |
| [`projects/13_portfolio_construction`](projects/13_portfolio_construction) | HRP, NCO, denoising | Reproduced; beats raw Markowitz worst-case 39/39, ties 1/N on Sharpe |
| [`projects/14_causal_factor_investing`](projects/14_causal_factor_investing) | Causal factor investing, backdoor | Reproduced; naive false-positive 100% to 5% corrected; 0/9 real factors survive |
| [`projects/15_sample_uniqueness_bootstrapping`](projects/15_sample_uniqueness_bootstrapping) | Sample uniqueness, sequential bootstrap | Reproduced; only about 41 to 44% of labels independent. An honesty fix |
| [`projects/16_meta_strategy_organization`](projects/16_meta_strategy_organization) | Meta-strategy assembly line | Reproduced over about 1.2M configs; lone best-IS pick gives pooled OOS Sharpe -0.02 |
| [`projects/15_cross_sectional_ml`](projects/15_cross_sectional_ml) | Cross-sectional ML rotation | Apparent survivor as first evaluated; audited to a leak in `apex_audit/` |
| [`apex_audit`](apex_audit) | Forensic audit of the lone survivor | Profit factor 2.17 to 1.08 on ablation; an artifact, clears honest deflation 0/10 |
| [`positive_control`](positive_control) | Acceptance-bar calibration | The gates discriminate: pass planted edges, reject noise, catch a planted leak |
| [`extensions`](extensions) | Eighteen practitioner extensions | None manufactured an edge; every discovery-pass positive was overturned |

## Data

No market data is bundled here. Every script reads from local data roots that you supply.
See [`DATA.md`](DATA.md) for the sources, schemas, and coverage (Binance USD-M perpetual
1m/1h/30m dumps, Algoseek ETF and equity bars, HistData FX quote ticks resampled to a
1-minute base). The at-scale strategy corpora behind the overfitting keystone (tens of
thousands of costed per-strategy daily PnL series per market) are produced by a separate
pipeline; the study scripts read the resulting per-strategy daily PnL from the
`LDP_PNL_DAILY` root.

## Setup and reproduce

Requires Python 3.10 or newer. Install the stack with `pip install -r requirements.txt`.

All data paths are centralized in [`config.py`](config.py), which reads each root from an
environment variable and falls back to a repo-relative default under `data/`. Point the
variables at your own copies of the datasets, for example:

```bash
export LDP_CRYPTO_1M=/path/to/binance_perp_1m       # *_1m.parquet
export LDP_CRYPTO_1H=/path/to/binance_perp_1h       # binance_um/*_1h.parquet
export LDP_CRYPTO_30M=/path/to/binance_perp_30m     # *_30m.parquet
export LDP_EQUITY_1M=/path/to/algoseek_etf_1min     # *.csv.gz
export LDP_FX_1M=/path/to/histdata_fx_1m            # *_fx1m.parquet
export LDP_PNL_DAILY=/path/to/pnl_daily             # per-strategy daily PnL corpus
```

Then run any study from the repository root, for example:

```bash
python3 projects/01_information_driven_bars/scripts/run_bars_multimarket.py
python3 positive_control/positive_control_v3.py
```

## Layout

```
config.py            central data-root configuration (env vars + relative defaults)
requirements.txt     Python dependencies
DATA.md              data sources, schemas, coverage
lib/                 shared code (bars, fractional differencing, deflation, costing, ...)
projects/NN_name/    one folder per core method: scripts + README
positive_control/    acceptance-bar calibration
apex_audit/          forensic audit of the lone cross-sectional survivor
extensions/          eighteen practitioner extensions under the same bar
```
