# Information-driven bars

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapter 2.

Time bars sample the market on the clock; information-driven bars sample it on activity. This project builds tick, volume, and dollar bars (and the advanced tier of imbalance and run bars) and tests López de Prado's claim that they push returns closer to IID-Gaussian than fixed time bars. It runs across three markets: crypto (Binance USD-M perp 1-minute dumps), US-equity ETFs (Algoseek 1-minute trade bars), and spot FX (HistData quote ticks, where tick count is the only available information clock). For equities it separates regular trading hours from extended hours so a session/overnight-gap artifact is not mistaken for a property of the method. A granularity study varies the base brick (1, 5, 15, 30, 60 minutes) while holding the target bar frequency fixed, to show how the advantage depends on how coarse the building blocks are.

Verdict: reproduced. Dollar, volume, and tick bars Gaussianize returns in all three markets, with excess kurtosis cut roughly 3 to 5 times. The size of the effect depends on bar granularity and on session handling.

## Run

From the repo root. Crypto cross-section from clean 1-minute data, multi-market crypto plus equities, equities session study, and forex:

```
python3 projects/01_information_driven_bars/scripts/run_bars_study_clean.py
python3 projects/01_information_driven_bars/scripts/run_bars_multimarket.py
python3 projects/01_information_driven_bars/scripts/run_equities_bars.py
python3 projects/01_information_driven_bars/scripts/run_forex_bars.py
```

Granularity sensitivity, the imbalance/run-bar extension, and the assembled multi-market summary:

```
python3 projects/01_information_driven_bars/scripts/run_granularity_study.py
python3 projects/01_information_driven_bars/scripts/run_imbalance_run_bars.py
python3 projects/01_information_driven_bars/scripts/make_final_figure.py
```

(`run_bars_study.py` is the earlier 30-minute-base version, kept for comparison.) Data roots come from `config.py` and its `LDP_*` environment variables: crypto scripts need `LDP_CRYPTO_1M` (or `LDP_CRYPTO_30M`), equities need `LDP_EQUITY_1M`, forex needs `LDP_FX_1M`. See `../../DATA.md` and `../../config.py`.
