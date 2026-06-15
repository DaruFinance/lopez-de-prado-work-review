# Data sources (reproducibility)

All results use real public market data. The scripts reference local cache
directories; point them at your own copies. No data is bundled in this repo.

- **Crypto**, Binance USD-M perpetual klines (open/high/low/close, volume,
  quote_volume, trade count, taker-buy volumes), downloaded from Binance's public
  data dumps (`data.binance.vision`). Coverage differs by study:
  - the multi-market 1-minute base used by the per-method studies is ~24 majors;
  - the backtest-overfitting corpus uses 20 perpetuals on 1m and 1h spanning
    **2018-05-06 to 2026-05-04**;
  - the cross-sectional-ML / apex-audit panel is the **survivorship-honest 787-perp
    1h universe** (live + delisted, `crypto_perp_delisted_1h`), 58,931 bars,
    **2019-09-08 to 2026-05-30**, with auxiliary funding / open-interest / order-flow
    (native 4h/8h) / spot feeds for the basis and order-flow features.
- **US Equities**, Algoseek ETF/equity 1-minute trade bars (SPY, QQQ, IWM, sector
  SPDRs, vol ETFs; ~250 large-caps for the daily cross-section), with trade count
  and VWAP, spanning **2007-01-03 to 2026-05-22**. Commercial dataset.
- **Forex**, HistData.com free quote ticks for 8 majors, resampled to a
  1-minute base with tick count (spot FX has no consolidated volume, so the
  information clock is tick count), spanning **2022-01-02 to 2024-12-31**.

All data roots are configured centrally in `config.py`, which reads each root
from an `LDP_*` environment variable and falls back to a repo-relative default
under `data/`. Point those variables at your own copies of the sources above; no
data is bundled here. See the "Setup & reproduce" section of the top-level
`README.md` for the variable names.
