# Purged k-fold, embargo, and combinatorial purged cross-validation

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapters 7 and 12.

Financial labels are built from windows of future bars, so consecutive labels share information. Standard k-fold cross-validation puts overlapping points on both sides of the train/test cut, leaking training information into the test fold and biasing the score upward. Purging removes train points whose label window overlaps the test fold; an embargo drops a few train points right after it to kill the residual serial correlation. Combinatorial purged cross-validation (CPCV) holds out every combination of test groups and returns a distribution of out-of-sample paths instead of one number. This project builds fixed-horizon overlapping labels on real bars (crypto dollar bars, US-equity ETF dollar bars, forex tick bars), trains a bagged-tree classifier on strictly-causal features, and scores it three ways: plain k-fold (leaky), purged k-fold with embargo (debiased), and CPCV (the full path distribution).

Verdict: reproduced. The naive k-fold leakage scales with the ratio of the label horizon to the fold size. The spread of CPCV paths is about 50 times the leakage bias that purging removes, so the choice of which single backtest you happened to run swamps the leakage correction.

## Run

From the repo root:

```
python3 projects/04_cross_validation/scripts/run_cv_leakage.py
```

A faster pass over fewer instruments:

```
python3 projects/04_cross_validation/scripts/run_cv_leakage.py --quick
```

Data roots come from `config.py` and its `LDP_*` environment variables: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M`. See `../../DATA.md` and `../../config.py`.
