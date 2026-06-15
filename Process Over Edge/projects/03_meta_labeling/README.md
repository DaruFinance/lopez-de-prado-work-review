# Triple-barrier labeling and meta-labeling

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapter 3; *Machine Learning for Asset Managers* (Cambridge, 2020), Chapter 5.

A primary model decides the side of a bet; a secondary meta-model decides whether to act on it and how large to size. This project implements both halves and tests whether the secondary adds return. For each instrument it builds dollar bars (crypto and equities) or tick bars (forex), takes an EMA-crossover primary, and labels each event with the triple-barrier method using full intrabar OHLC to find the first barrier touched (profit-take, stop-loss, or max-holding). The meta-label marks whether the primary's bet finished in profit. A bagged-tree secondary is trained and scored with purged k-fold cross-validation so overlapping labels do not leak across folds, and the meta-gated strategy is compared to primary-only on out-of-fold PnL net of per-turnover costs. The headline metric is the Deflated Sharpe Ratio, with PBO and effective-N alongside; it runs across 42 instruments spanning all three markets.

Verdict: failed as an alpha source. The meta-model lifts profit factor in 38 of 42 instruments, but 0 of 42 clear the Deflated Sharpe bar. Meta-labeling is a precision and filtering tool, not a source of return. (Note the precondition this run violates: it filters a primary with no established edge. The sibling project `03b_metalabel_real_primary` redoes the test with that precondition met.)

## Run

From the repo root:

```
python3 projects/03_meta_labeling/scripts/run_meta_labeling.py
```

A fast sanity pass:

```
python3 projects/03_meta_labeling/scripts/run_meta_labeling.py --smoke
```

(`tbm.py` is the triple-barrier and meta-labeling library imported by the driver, not run directly.) Data roots come from `config.py` and its `LDP_*` environment variables: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M`. See `../../DATA.md` and `../../config.py`.
