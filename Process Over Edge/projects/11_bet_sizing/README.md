# Bet Sizing (López de Prado, AFML Ch.10)

This study sizes bets from predicted probabilities and asks whether sizing adds
risk-adjusted edge or only cuts turnover. Across 42 real instruments (27 crypto
perps, 7 US-equity ETFs, 8 FX pairs) with 32 IS-tunable trials each, it compares
three books on the same predictions: fixed size, probability size, and
discretized-probability size. A bagged-tree model produces the probabilities; the
books are scored net of costs. The headline metric is the Deflated Sharpe Ratio,
with PBO and effective-N as deflation diagnostics. A deepening pass works off the
per-instrument result frame to separate the turnover effect from the edge effect,
reporting paired deltas and by-market sign tests.

## Verdict

Failed as alpha. Probability sizing cuts turnover 80 to 87% but adds no
deflation-survivable edge: 0 of 42 instruments clear the Deflated Sharpe bar. It
is a cost and turnover tool, not a source of return.

## Run

From the repo root:

```bash
# full 42-instrument run, single process
bash projects/11_bet_sizing/run_full.sh

# or directly
python3 projects/11_bet_sizing/scripts/run_bet_sizing.py

# separate the turnover effect from the edge effect
python3 projects/11_bet_sizing/scripts/deepen.py
```

`run_bet_sizing.py` also accepts `--smoke`, `--profile`, and `--verify-kernel`.

## Data

Needs the `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, and `LDP_FX_1M` roots. See
[../../DATA.md](../../DATA.md) for the sources and [../../config.py](../../config.py)
for the environment variables and defaults.
