# Meta-strategy assembly-line organization and mandatory trial disclosure

López de Prado, *Advances in Financial Machine Learning*, Ch.1 and Ch.16 (the research "production line" and the multiple-testing discipline).

This study applies the assembly-line organization and the trial-disclosure requirement to the program itself, run across about 1.2 million strategy configurations from the real net-daily-PnL corpus. The centerpiece compares two research processes head to head over a rolling walk-forward. The lone backtester picks, in every window, the single strategy with the best in-sample Sharpe and deploys it with no disclosure of the trial count and no deflation. The disciplined assembly line uses the same candidate pool but gates the in-sample winners by the Deflated Sharpe Ratio against the full disclosed trial count, deploys only the survivors sized by an HRP allocation, and holds cash when nothing survives. Supporting experiments cover the meta-strategy portfolio (combining many weakly-correlated bets and deflating the sleeve at the portfolio level), program-level PBO via CSCV, and the expected-maximum Sharpe of the program against its observed best using real dispersion and real effective-N. A capstone chains every station of the production line (dollar bars, triple-barrier labels, uniqueness weights, a meta-labeling classifier, bet sizing, HRP allocation) into one disclosed pipeline run on a never-touched out-of-time window. All PnL is real and after-cost; the only sanctioned synthetic component is the skill-less Monte-Carlo that validates the expected-max-Sharpe formula.

## Verdict

Reproduced. Run across about 1.2 million configurations, the disciplined gate deploys essentially nothing. The simulated lone backtester, picking the best in-sample run, ships a strategy whose pooled out-of-sample Sharpe is -0.02. The winner's curse, made concrete.

## Run

From the repository root, each experiment is a standalone script (no single wrapper):

```
python3 projects/16_meta_strategy_organization/scripts/e1_sisyphus_vs_assembly.py   # lone backtester vs disclosed gate
python3 projects/16_meta_strategy_organization/scripts/e2_meta_portfolio.py         # diversified sleeve, portfolio-level DSR
python3 projects/16_meta_strategy_organization/scripts/e3_program_pbo.py            # program PBO via CSCV
python3 projects/16_meta_strategy_organization/scripts/e4_emax_real.py              # E[max Sharpe] vs observed best
python3 projects/16_meta_strategy_organization/scripts/e5_full_pipeline.py          # full assembly line, out-of-time window
python3 projects/16_meta_strategy_organization/scripts/exp_max_sharpe.py            # expected-max-Sharpe curve + MC check
python3 projects/16_meta_strategy_organization/scripts/meta_analysis.py             # cross-study scorecard
python3 projects/16_meta_strategy_organization/scripts/make_figures.py              # result figures
```

Data roots needed: `LDP_PNL_DAILY` (the per-strategy daily-PnL corpus, used by the e1-e4 experiments and the scorecard) and `LDP_CRYPTO_1M` (1-minute base bars for the e5 capstone pipeline). `meta_analysis.py` also reads the result tables produced by the other projects. All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
