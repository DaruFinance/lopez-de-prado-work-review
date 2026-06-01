# Data sources (reproducibility)

All results use real public market data. The scripts reference local cache
directories; point them at your own copies. No data is bundled in this repo.

- **Crypto**, Binance USD-M perpetual 1-minute klines (open/high/low/close,
  volume, quote_volume, trade count, taker-buy volumes), downloaded from
  Binance's public data dumps (`data.binance.vision`). 2022-2024, ~24 majors.
- **US Equities**, Algoseek ETF 1-minute trade bars (SPY, QQQ, IWM, sector
  SPDRs, vol ETFs), with trade count and VWAP. Commercial dataset.
- **Forex**, HistData.com free quote ticks for 8 majors, resampled to a
  1-minute base with tick count (spot FX has no consolidated volume, so the
  information clock is tick count).

All data roots are configured centrally in `config.py`, which reads each root
from an `LDP_*` environment variable and falls back to a repo-relative default
under `data/`. Point those variables at your own copies of the sources above; no
data is bundled here. See the "Setup & reproduce" section of the top-level
`README.md` for the variable names.
