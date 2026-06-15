# Structural Breaks and Entropy Features (López de Prado, AFML Ch.17-18)

This study implements structural-break tests and entropy features, then asks
whether either predicts direction after realistic costs. It builds matched dollar
bars per instrument, takes causal close-to-close log returns, and computes
backward-only features: a CUSUM event sampler, SADF explosiveness on a rolling
capped backward window, and rolling Shannon, Lempel-Ziv, and Kontoyiannis entropy
of the binary return string. Each feature gets a per-feature sign rule fit in
fold, scored under purged k-fold CV with an embargo, net of costs. The universe
is 27 real instruments across crypto, equities, and forex. The headline metric is
the Deflated Sharpe Ratio with CSCV PBO over the feature corpus. A deepening pass
breaks the result down by feature family and by market, samples on CUSUM events
rather than the raw clock, and runs sign tests per family.

## Verdict

Failed on direction. The Deflated Sharpe Ratio is about 0.002, AUC sits near a
coin flip, and the costed sign rule is net-negative in 22 of 24 cases. The break
and entropy features describe regimes but carry no tradable directional signal.

## Run

From the repo root:

```bash
# kernel parity check, then the full multi-market run
bash projects/07_structural_breaks_entropy/run_full.sh

# or directly (default label is fixed-horizon)
python3 projects/07_structural_breaks_entropy/scripts/run_breaks_entropy.py --verify
python3 projects/07_structural_breaks_entropy/scripts/run_breaks_entropy.py --label fixed

# family-by-market breakdown and CUSUM-event sampling
python3 projects/07_structural_breaks_entropy/scripts/deepen_breaks_entropy.py
```

`run_breaks_entropy.py` also accepts `--smoke`, `--profile`, and
`--label trendscan`.

## Data

Needs the `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M` roots. See
[../../DATA.md](../../DATA.md) for the sources and [../../config.py](../../config.py)
for the environment variables and defaults.
