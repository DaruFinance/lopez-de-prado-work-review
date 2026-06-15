# Ensembles and Feature Importance (López de Prado, AFML Ch.6 and Ch.8)

This study compares bagging against boosting and compares MDI feature importance
against MDA. For 42 real instruments (27 crypto dollar bars, 7 ETF dollar bars, 8
FX tick bars), it fits a bagged RandomForest (low max-features, a nonzero
min-weight-fraction-leaf, uniqueness sample weights, max-samples set to average
label uniqueness) against a HistGradientBoosting model, each hyper-tuned over a
purged grid by negative log-loss with accuracy as a control. It reports the
out-of-sample Deflated Sharpe Ratio and the in-sample to out-of-sample log-loss
gap for each. Feature importance covers MDI, MDA, and clustered MDA (permutation,
log-loss scored, out of sample), the MDI substitution-bias signal, and top-set
stability across CPCV paths by Jaccard. Everything is causal, costed, and scored
under purged CV / CPCV. A deepening pass pairs the bagging-versus-boosting gap
across instruments and adds clustered importance as the substitution-effect
remedy.

## Verdict

Reproduced. Bagging generalizes about 4.8x tighter on the in-sample to
out-of-sample gap than boosting on 100% of 40 instruments. MDI is
substitution-biased; MDA is not.

## Run

From the repo root:

```bash
# full run, 16-way instrument-level pool
bash projects/09_ensembles_importance/run_full.sh

# or directly
python3 projects/09_ensembles_importance/scripts/run_ensembles_importance.py --jobs 16

# paired bagging-vs-boosting and importance breakdown
python3 projects/09_ensembles_importance/scripts/deepen.py --jobs 8

# clustered feature importance (substitution-effect remedy)
python3 projects/09_ensembles_importance/scripts/clustered_importance.py
```

`run_ensembles_importance.py` also accepts `--smoke`, `--profile`, `--verify`,
and `--light`.

## Data

Needs the `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M` roots. See
[../../DATA.md](../../DATA.md) for the sources and [../../config.py](../../config.py)
for the environment variables and defaults.
