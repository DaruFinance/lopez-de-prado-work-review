# Fractional differentiation

Reference: Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Chapter 5.

Price levels carry memory but fail stationarity tests; plain returns are stationary but throw the level information away. Fixed-Width Window Fractional Differentiation (FFD) looks for the middle: the smallest fractional differencing order that makes a log-price series pass an augmented Dickey-Fuller test while still tracking the price level. This project implements the causal FFD transform, searches the differencing grid for the minimum stationary order on each series, and measures how much memory survives at that order versus at plain differencing. It runs on the full Binance USD-M perp cross-section at 1-hour granularity, then extends the same pipeline (identical d grid, tau, ADF logic, and memory metric) to US-equity ETFs and forex so the order is comparable across markets. A frequency study checks whether the order is a property of the price process or of the sampling rate by comparing 1-minute and 1-hour bases.

Verdict: reproduced and universal. At the order where the series first becomes stationary, the fractionally-differenced series keeps about 0.98 correlation with the price level, against about 0.01 for plain returns. Stationarity does not have to cost the memory.

## Run

From the repo root. Crypto reproduction and cross-section, the multi-market extension, and the frequency-sensitivity study:

```
python3 projects/02_fractional_differentiation/scripts/run_fracdiff_study.py
python3 projects/02_fractional_differentiation/scripts/run_fracdiff_multimarket.py
python3 projects/02_fractional_differentiation/scripts/run_frequency_study.py
```

Run `run_fracdiff_study.py` first; the multi-market script reuses its banked crypto per-pair results and only computes equities and forex. Data roots come from `config.py` and its `LDP_*` environment variables: the crypto runs need `LDP_CRYPTO_1H` and `LDP_CRYPTO_1M`, the multi-market run also needs `LDP_EQUITY_1M` and `LDP_FX_RAW`. See `../../DATA.md` and `../../config.py`.
