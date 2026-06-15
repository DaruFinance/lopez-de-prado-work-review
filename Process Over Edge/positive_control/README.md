# Positive control: does the acceptance bar accept a true edge?

A stack of negative results is only informative if the bar that produced them can still
recognise a genuine edge. These scripts measure the bar's false-negative behaviour
directly, on real data and on controlled data.

## Real-data control (`positive_control_real.py`)

This is the headline calibration. It plants a tradeable edge of controllable size into
real Binance BTCUSDT 30-minute returns (110,975 return bars, 6.3 years), charges the
crypto cost model, evaluates strictly walk-forward over quarterly windows, and sweeps the
edge size to find the smallest one the full acceptance bar accepts. The real returns
supply the risk; a persistent, causal position with no market edge of its own carries the
real volatility, and a constant per-bar drift is added as the planted skill. The no-edge
case is the same position and turnover with no drift.

The bar is judged against two benchmarks. As a single pre-specified hypothesis the
deflation benchmark is zero. At the corpus search size the deflation benchmark is the
False-Strategy-Theorem expected maximum for the 50,000-strategy crypto search, an
annualised Sharpe of 2.72.

### Result

| Quantity | Value |
|---|---|
| No-edge strategy pass rate | 0.01 as a single hypothesis, 0.00 at corpus scale |
| Single-hypothesis detection floor | net annualised Sharpe about 0.67 (median over 200 realisations) |
| Corpus-scale detection floor | net annualised Sharpe about 3.38 (deflation against the 2.72 FST maximum) |
| Best strategy found in the real corpus | 2.0, which sits below the corpus floor |

The binding gate at the single-hypothesis level is sign-consistency, a majority of
quarters positive; the deflation and tail-guard gates clear below it. At corpus scale the
binding gate is deflation. The bar accepts a genuine persistent edge and rejects the
no-edge strategy, so the negative results elsewhere are calibrated rather than an artifact
of an over-strict bar, and the best strategy actually found (2.0) is below the
search-size floor.

### Run

```
export LDP_CRYPTO_30M=/path/to/binance_perp_30m   # directory holding BTCUSDT_30m.parquet
python3 positive_control/positive_control_real.py
```

It reads only the one BTCUSDT 30-minute file (see `../DATA.md` and `../config.py`), prints
the floors, and writes its result JSON next to the script.

## Synthetic complement (`positive_control_v3.py`)

A controlled-data check that the gates discriminate by construction. It builds return
panels with a planted edge of known strength, sweeps the strength from zero through
strong, and adds a return-additive leak as a fifth case. Pure noise is rejected, moderate
and strong edges pass, the marginal case is adjudicated rather than waved through, and the
planted leak is caught by the causality test despite clearing the economic gates. It
generates its panels internally and needs no market-data root. Versions v1 and v2 are
kept for provenance.

The harder forward-fill leak, where a coarse-frequency feature survives a naive
pollute-and-verify, is demonstrated and caught separately in [`../apex_audit/`](../apex_audit/)
through the same-bar-versus-future correlation and feature ablation.

```
python3 positive_control/positive_control_v3.py
```
