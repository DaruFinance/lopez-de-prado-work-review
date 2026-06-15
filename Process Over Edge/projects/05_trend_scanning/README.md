# Trend-Scanning Labels (López de Prado, ML4AM Ch.5)

This study builds the trend-scanning labeller (label = sign of the maximum-|t|
forward OLS slope over a band of horizons, with |t| as the confidence/size) and
compares it against a fixed-horizon control labeller. The labels are
forward-looking targets only. A bagged-tree secondary model then predicts the
label sign from causal features under purged k-fold CV with an embargo, and the
strategy trades the out-of-fold predicted side over a causal forward hold, net of
per-side costs (crypto 7 bp, equities 2 bp, forex 1 bp). The run spans 42 real
instruments across three markets (27 Binance USD-M perp dollar bars, 7 Algoseek
ETF dollar bars, 8 HistData FX tick bars). The headline metric is the Deflated
Sharpe Ratio over the 9-trial IS-tunable grid, with PBO (CSCV) and effective-N as
deflation diagnostics. A deepening pass adds a triple-barrier target on the same
event set so all three labellers are compared head to head.

## Verdict

Failed as alpha. The profit-factor win for trend-scan over fixed-horizon shows up
in only 28 of 42 instruments and is deflation-neutral. What looks like an edge is
horizon length, not the signal: the labeller's advantage is the hold it implies,
not a better side or size.

## Run

From the repo root:

```bash
# kernel parity check, then the full 42-instrument run
bash projects/05_trend_scanning/run_full.sh

# or step by step
python3 projects/05_trend_scanning/scripts/run_trend_scanning.py --verify
python3 projects/05_trend_scanning/scripts/run_trend_scanning.py

# trend-scan vs fixed-horizon vs triple-barrier on a shared event set
python3 projects/05_trend_scanning/scripts/deepen_labels.py
```

Smoke and profile modes are available via `--smoke` and `--profile` on
`run_trend_scanning.py`.

## Data

Needs the `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M` roots. See
[../../DATA.md](../../DATA.md) for the sources and [../../config.py](../../config.py)
for the environment variables and defaults.
