# Data sources

All results use real market data. No data is bundled in this repository. Each script
reads from local roots that you supply, configured centrally in `config.py`: every root
resolves from an `LDP_*` environment variable and falls back to a repo-relative default
under `data/`. Point the variables at your own copies of the sources below.

## Sources

- **Crypto**: Binance USD-M perpetual klines (open/high/low/close, volume, quote volume,
  trade count, taker-buy volumes) from the public Binance data dumps
  (`data.binance.vision`), at 1-minute, 30-minute, and 1-hour granularity. The
  cross-sectional study uses a survivorship-honest panel of 787 perpetuals at 1 hour,
  including delisted pairs carried only over the bars they actually traded. Auxiliary
  per-pair feeds (8-hour funding, open interest, taker order flow, COIN-M inverse perps)
  back the cross-sectional rotation and its audit.
- **US equities and ETFs**: Algoseek ETF and equity bars (SPY, QQQ, IWM, sector SPDRs, vol
  ETFs, and a large-cap cross-section), with trade count and VWAP. Commercial dataset.
- **FX**: HistData.com free quote ticks for 8 majors, resampled to a 1-minute base with
  tick count. Spot FX has no consolidated volume, so the information clock is tick count.

## Derived corpora

The at-scale strategy corpora behind the overfitting keystone (tens of thousands of
costed, per-strategy daily PnL series per market) and the banked cross-sectional run
directories are produced by separate pipelines and are not part of this repository. The
study scripts read the resulting per-strategy daily PnL from `LDP_PNL_DAILY` and the
banked cross-sectional ledgers from the `LDP_XS_*` roots. The per-instrument figures
regenerate end to end from the raw price roots.

## Environment variables

See `config.py` for the full list and the default layout each root expects. The most
commonly needed are `LDP_CRYPTO_1M`, `LDP_CRYPTO_1H`, `LDP_CRYPTO_30M`,
`LDP_CRYPTO_PERP_1H`, `LDP_CRYPTO_DELISTED_1H`, `LDP_EQUITY_1M`, `LDP_EQUITY_DAILY_XS`,
`LDP_FX_1M`, and `LDP_PNL_DAILY`.
