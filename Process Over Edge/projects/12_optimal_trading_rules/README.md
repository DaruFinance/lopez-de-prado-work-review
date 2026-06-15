# Optimal trading rules on an Ornstein-Uhlenbeck process, plus the Triple-Penance drawdown rule

López de Prado, *Advances in Financial Machine Learning*, Ch.13; Bailey & López de Prado, "Determining Optimal Trading Rules without Backtesting" (2014) and "The Triple Penance Rule" / stop-outs under serial correlation (2015).

This study runs across 42 instruments on real 1-minute data: 27 crypto perpetuals, 7 US-equity ETFs, and 8 forex pairs. For each instrument it builds dollar bars (crypto and equities) or tick bars (forex), forms a causal z-scored mean-reversion level series, and fits an OU/AR(1) process on the in-sample slice. The optimal (profit-take, stop-loss) rule comes from a Monte-Carlo mesh on the fitted process, with parameters fit only to in-sample data so there is no lookahead. The rule is then validated on real out-of-sample bars using full intrabar-OHLC first-touch exits and per-side costs (crypto 7 bp, equities 2 bp, forex 1 bp, charged on entry and exit). The headline is the Deflated Sharpe Ratio of the OU rule against an in-sample-tuned fixed PT/SL control, with the 27 tuned trials per instrument forming the trial set, plus PBO and effective-N. The Triple-Penance section measures the AR(1) coefficient of the out-of-sample strategy returns and reports the serial-correlation-aware maximum-drawdown and time-under-water bounds against the naive IID figures.

## Verdict

Partial reproduction. The OU optimal-trading rule on a single series performs at about a coin flip out of sample, so the trading rule itself is not a validated edge. The Triple-Penance maximum-drawdown rule does reproduce: the IID maximum-drawdown understates the true serial-correlation-aware drawdown by roughly 3.2x. The drawdown formula is the piece that holds up; the OU trading rule is not.

## Run

From the repository root:

```
python3 projects/12_optimal_trading_rules/scripts/run_optimal_trading_rules.py
```

Useful flags: `--smoke` (one instrument per market, tiny Monte-Carlo), `--profile` (cProfile a single instrument), `--verify` (Numba bit-identical kernel check only). The wrapper `projects/12_optimal_trading_rules/run_full.sh` pins threads for a deterministic single-core run.

Data roots needed: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, `LDP_FX_1M` (and `LDP_CRYPTO_1H`). All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
