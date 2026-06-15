# Forensic audit of the lone cross-sectional ML survivor

This directory isolates the one strategy that survived the wider screen: a
cross-sectional gradient-boosted-tree rotation over a 787-perpetual hourly panel.
Each bar ranks the alive universe by a causal per-pair score and rotates capital
into the top and bottom quantiles, market-neutral. The audit rebuilds that engine
from scratch and stress-tests it: structural controls (label permutation, label
time-shift, and a pollute-and-verify poison test), a same-moment-versus-forward
correlation pre-screen on the cached panel, and feature ablation. The panel is
survivorship-honest, including delisted pairs gated at their delisting bar and a
`volume > 0` mask that excludes phantom stale-price bars.

## Verdict

Two known defects were fixed first: a one-bar return lookahead (`ret[t]` replaced by
`ret_lag1`) and a survivorship leak (the `volume > 0` validity mask). After both fixes
the edge got *stronger*, which is the signature of a third dominant leak rather than a
real signal. Every structural control passed. Feature ablation then collapsed the
shortest-horizon profit factor from 2.17 to 1.08, and to 0.95 (an outright loss) at the
one-day horizon. The order-flow feature is the cause. It is sourced from coarse 4h/8h
bars, forward-filled onto the 1h grid, then lagged one bar, yet it still overlaps the
booked return: consecutive bars carry an identical value about 81% of the time, so a
one-bar lag does not decontaminate it. On honest accounting the strategy clears
deflation at 0 of 10 horizons. The lone survivor was a data-handling artifact. Deflation
is blind to leakage, so a trial-count penalty has to be paired with feature ablation.

## How to run

Run each script from the repository root. Resolve the data roots first; see
`../DATA.md` and `../config.py` for the full list and the `LDP_*` overrides.

    python3 apex_audit/run_clean_audit.py     # pollute test, then the full clean lgbm WFO
    python3 apex_audit/s12_battery.py         # EDGE-bar robustness battery on the clean OOS ledger
    python3 apex_audit/s12_leak_tiebreak.py   # same-moment vs forward correlation tiebreak

`run_clean_audit.py` writes its ledger under `apex_audit/runs/` and the cached panel under
`apex_audit/_cache/`; the other two scripts read those outputs. `xs_stats.py` produces the
de Prado overfitting report over banked run dirs (`python3 apex_audit/xs_stats.py
[run_dir ...]`).

### Data roots

Set these to your own copies of the sources described in `../DATA.md`:

- `LDP_CRYPTO_PERP_1H` — Binance USD-M perpetual hourly klines (live universe).
- `LDP_CRYPTO_DELISTED_1H` — point-in-time delisted perpetual hourly klines.
- `LDP_CRYPTO_SPOT_1H` — spot hourly klines for the perp underlyings (basis).
- `LDP_CRYPTO_LISTING_DATES` — per-instrument listing/delisting dates CSV.
- `LDP_STRUCTURAL_ENGINE` — structural-edge engine supplying the verified cost constants.
- `LDP_CRYPTO_FUNDING`, `LDP_CRYPTO_OI`, `LDP_CRYPTO_ORDERFLOW` — funding, open-interest
  and order-flow feeds (the last is the feature this audit indicts).
- `LDP_XS_RUNS` — banked cross-sectional run dirs read by `xs_stats.py`.

Outputs (`runs/`, `_cache/`) land under this directory.
