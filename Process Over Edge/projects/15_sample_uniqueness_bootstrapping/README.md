# Sample uniqueness and the sequential bootstrap

López de Prado, *Advances in Financial Machine Learning*, Ch.4.

Overlapping triple-barrier labels are not independent: a label spanning bars [t0, t1] shares information with every other label whose span overlaps it. This study reproduces López de Prado's apparatus at scale on real bars across crypto, US equities, and forex. Per instrument it builds dollar bars (crypto and equities) or tick bars (forex), forms events from an EMA crossover, triple-barriers each event with full intrabar OHLC, then computes label concurrency, average uniqueness, and the effective sample size (the sum of label uniqueness). It compares the sequential bootstrap against a standard IID bootstrap on the average uniqueness of the drawn samples, and tests whether the correction matters for a model: bagged trees trained naively (IID bags, no weights) versus corrected (sequential-bootstrap bags, max_samples set to average uniqueness, return-attribution sample weights), compared on out-of-sample neg-log-loss and AUC under purged k-fold with embargo. A companion script links the honest effective sample size to the Probabilistic and Deflated Sharpe Ratios and the Minimum Track Record Length. The Numba concurrency and bootstrap kernels are verified bit-identical to a pure-Python reference, and the implementation never materializes a dense T-by-N indicator matrix.

## Verdict

Reproduced. Overlapping labels are non-IID, with an effective sample size of only about 41 to 44 percent of N across the three markets. The sequential bootstrap raises average uniqueness in 42 of 42 instruments. This is an honesty and robustness correction, not a source of return: 0 of 42 instruments improve out-of-sample accuracy.

## Run

From the repository root:

```
python3 projects/15_sample_uniqueness_bootstrapping/scripts/run_uniqueness.py
```

Flags: `--smoke` (one instrument per market), `--profile` (profile a single instrument). The deflation link and the kernel verification run separately:

```
python3 projects/15_sample_uniqueness_bootstrapping/scripts/uniqueness_deflation.py   # uniqueness -> DSR/PSR/MinTRL
python3 projects/15_sample_uniqueness_bootstrapping/scripts/verify.py                 # Numba vs pure-Python bit-identity
```

Data roots needed: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, `LDP_FX_1M`. All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
