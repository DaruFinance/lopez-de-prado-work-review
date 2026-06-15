# Cross-sectional machine-learning rotation

Gu, Kelly & Xiu cross-sectional return-prediction setting, scored on the program's standard López de Prado deflation apparatus (Deflated Sharpe Ratio, PBO, effective-N; *Advances in Financial Machine Learning*, Ch.8).

This folder holds the cross-sectional ML rotation as it was originally evaluated. Each bar ranks a wide crypto-perpetual universe by a causal per-pair tree score and rotates capital long the top quantile and short the bottom quantile, market-neutral, with rotation knobs tuned in-sample per purged walk-forward window and realistic per-fill costs charged. Per-horizon out-of-sample profit factor, Sharpe, and Deflated Sharpe are scored against the False Strategy Theorem benchmark, with the horizon set treated as the trial family. A US-equity large-cap cross-section is run to the same recipe as a cross-market check, and gradient-boosted-tree and neural families are folded into one family-level deflation table. As originally evaluated this was the program's one apparent deflation survivor: gradient-boosted trees cleared the Deflated Sharpe bar at 10 of 10 horizons, PBO 0.10, in-sample-to-out-of-sample rank rho +0.78, median out-of-sample profit factor 1.123, with 23 of 30 tree combos clearing the False Strategy Theorem versus 0 of about 25,162 single-series strategies.

## Verdict

Not a real edge. This survivor was later audited from scratch and traced to a coarse-frequency order-flow feature that was forward-filled onto a finer grid, which leaks the contemporaneous return: consecutive bars carry an identical value about 81% of the time, so a one-bar lag does not decontaminate it. Feature ablation collapses the shortest-horizon profit factor from 2.17 to 1.08, and to 0.95 at the one-day horizon. On honest accounting the strategy clears deflation at 0 of 10 horizons. Deflation is blind to leakage, so a trial-count penalty has to be paired with feature ablation. Read this folder as the original evaluation, then see `../../apex_audit/` for the forensic teardown and the final verdict (an artifact, not an edge).

## Run

From the repository root:

```
python3 projects/15_cross_sectional_ml/scripts/run_cross_sectional_ml.py   # DSR/PBO by regime
python3 projects/15_cross_sectional_ml/scripts/run_family_dsr.py           # tree-family deflation table
python3 projects/15_cross_sectional_ml/scripts/run_nn_families.py          # fold in neural families
python3 projects/15_cross_sectional_ml/scripts/run_us_equity_xsection.py   # US-equity cross-market check
```

These scripts read banked per-bar ledgers and saved Sharpe/PF values produced by a separate training pipeline; they do not retrain models. `fetch_us_equity_daily.py` downloads the US-equity daily panel (API connection read from the environment). Data and engine roots needed: `LDP_XS_LEDGER`, `LDP_XS_NN_LEDGER`, `LDP_XS_MULTI`, `LDP_SINGLE_SERIES`, `LDP_XS_ENGINE`, and `LDP_EQUITY_DAILY_XS`. All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
