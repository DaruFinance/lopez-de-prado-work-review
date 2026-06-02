# US-equity cross-sectional ML rotation: per-horizon OOS and DSR

Universe: 250 large-cap US-listed names (predominantly S&P 500 members), adjusted daily bars 2015-01-02..2026-05-29, median 245 alive per bar. Family: lgbm. Market-neutral long-short rotation, purged WFO (IS=500/walk=150/sel=120 days), realistic per-fill costs (median 3.4 bp/side).

|   H |   oos_bars |   oos_pf |   oos_sharpe |   oos_sharpe_ann |   skew |   kurt |   sr0_benchmark |    dsr | dsr_survives   |
|----:|-----------:|---------:|-------------:|-----------------:|-------:|-------:|----------------:|-------:|:---------------|
|   1 |       2250 |   1.0958 |       0.0277 |            0.44  | -0.863 | 24.971 |          0.0133 | 0.7494 | False          |
|   5 |       2250 |   1.1051 |       0.0332 |            0.527 |  0.175 | 11.581 |          0.0133 | 0.8272 | False          |
|  10 |       2250 |   1.149  |       0.0466 |            0.74  |  0.521 | 13.277 |          0.0133 | 0.9443 | False          |
|  21 |       2250 |   1.1316 |       0.0413 |            0.656 |  0.152 | 12.698 |          0.0133 | 0.9077 | False          |
|  63 |       2250 |   1.06   |       0.0182 |            0.289 | -0.265 | 19.213 |          0.0133 | 0.5905 | False          |

## Multi-market comparison: crypto vs US equities

| market       | universe            | family   |   median_oos_pf |   median_oos_sharpe_ann | dsr_survive    |   pbo |
|:-------------|:--------------------|:---------|----------------:|------------------------:|:---------------|------:|
| crypto perps | 787 perp pairs      | lgbm     |          1.123  |                   2.66  | 10/10 horizons |   0.1 |
| US equities  | 250 large-cap names | lgbm     |          1.1051 |                   0.527 | 0/5 horizons   | nan   |

**Survivorship caveat.** The universe is a fixed list of CURRENT large-cap members, so names that were large-cap earlier but have since been removed (delistings, takeovers, demotions) are absent. This is a mild upward bias on the long side; the result is survivorship-aware but not survivorship-free. Later IPOs enter the cross-section only once they list (within-live-span gating), so there is no look-ahead onto pre-listing dates.

**Verdict.** The cross-sectional ML edge does NOT replicate in US equities: median OOS PF 1.105 (crypto 1.123), DSR survival 0/5 horizons.
