# Causal factor investing and the backdoor adjustment

López de Prado, "Causal Factor Investing" (2023) and the association-versus-causation critique; Pearl's structural causal models (fork, chain, collider).

This study makes the causal-inference critique of factor investing testable. The Monte-Carlo half builds the three elementary causal structures, a fork (common confounder), a chain (mediator), and a collider, and shows what a naive OLS factor regression reports under each: whether it is biased, whether the correct backdoor adjustment fixes it, and whether conditioning on the wrong variable manufactures significance. A dose-response sweep traces decision error against the strength of the structural bias. The real-data half runs cross-asset factors (momentum, realized vol, short-reversal) on daily data across crypto, US-equity ETFs, and forex, comparing a naive associational panel regression against two confounder-adjusted backdoor regressions and reporting where the significance verdict flips. The Deflated Sharpe Ratio is the headline metric on the naive long-short sleeves, and a hierarchy-of-evidence falsification checklist is applied to the survivors. All evidence is observational, factors are lagged with no lookahead, and no synthetic price data is used; the only simulated component is the structural Monte-Carlo, which has a known data-generating process.

## Verdict

Reproduced exactly. Under a hidden confounder the naive factor regression rejects the null 100% of the time; the backdoor adjustment restores the nominal 5% rejection rate. Conditioning on a collider manufactures significance where none exists. On the real factors, 0 of 9 survive correction and deflation.

## Run

From the repository root:

```
python3 projects/14_causal_factor_investing/scripts/run_causal.py --n-sims 20000 --n-obs 2000
```

Flags: `--smoke` (fast reduced run), `--profile` (profile the Monte-Carlo engine), `--n-sims` / `--n-obs` to set the simulation size. A natural-experiment extension built on Binance perpetual-futures listings is in `scripts/run_event_study.py` (`--smoke` for a quick pass). The wrapper `projects/14_causal_factor_investing/run_full.sh` runs the main study with the headline settings.

Data roots needed: `LDP_CRYPTO_1M`, `LDP_EQUITY_1M`, `LDP_FX_1M` for the main study, and `LDP_CRYPTO_PERP_1H`, `LDP_CRYPTO_SPOT_1H`, `LDP_CRYPTO_LISTING_DATES` for the event-study extension. All roots resolve through `config.py` from `LDP_*` environment variables; see `../../DATA.md` for the sources and `../../config.py` for the variable names and defaults.
