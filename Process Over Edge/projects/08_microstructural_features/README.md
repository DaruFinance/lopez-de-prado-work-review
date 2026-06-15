# Microstructural Features (López de Prado, AFML Ch.19)

This study builds Ch.19 microstructure and order-flow features, including
Bulk-Volume Classification, and tests whether they predict the next bar after
costs. For 24 real instruments across three markets (10 crypto perps, 7 US-equity
ETFs, 7 forex majors), it constructs information-driven bars (crypto and equity
dollar bars, forex tick bars), then scores next-bar direction and volatility AUC
plus a costed long/short Sharpe under purged k-fold CV with an embargo. The
headline metric is the Deflated Sharpe Ratio with PBO. Data tiers are handled
honestly: crypto has true signed taker volume, equities have volume and trade
count only (so order-flow estimators use the BVC proxy), and forex has tick count
only (price-only estimators plus a tick-rule Kyle). A separate completeness
script tests stationary versus raw order-flow presentation, and a deepening pass
isolates which estimator carries signal and how much the data tier matters.

## Verdict

Failed on direction. There is one hygiene win: the free BVC order-flow proxy
matches paid signed-flow data on the prediction task, so the result is about data
tier, not about return. The features do not produce a costed directional edge.

## Run

From the repo root:

```bash
# kernel parity check, then the full run (RandomForest uses all cores)
bash projects/08_microstructural_features/run_full.sh

# or directly
python3 projects/08_microstructural_features/scripts/run_microstructure.py --verify
python3 projects/08_microstructural_features/scripts/run_microstructure.py --jobs -1

# free BVC proxy vs paid signed flow, and stationary vs raw presentation
python3 projects/08_microstructural_features/scripts/stationary_vs_raw_flow.py
python3 projects/08_microstructural_features/scripts/deepen_microstructure.py
```

`run_microstructure.py` also accepts `--smoke` and `--profile`.

## Data

Needs the `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M` roots. See
[../../DATA.md](../../DATA.md) for the sources and [../../config.py](../../config.py)
for the environment variables and defaults.
