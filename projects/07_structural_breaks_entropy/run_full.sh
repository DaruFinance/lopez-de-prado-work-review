#!/usr/bin/env bash
# =============================================================================
# Project 7, Structural Breaks & Entropy Features (LdP AFML Ch.17-18)
# FULL multi-market run.
#
#   Markets / universe (>=10 per market where available):
#     crypto   : 12 Binance USD-M perps (BTC ETH SOL XRP DOGE BNB LINK AVAX LTC
#                ATOM BCH DOT), full 1m history (~1.5M base bars each).
#     equities : 7 Algoseek ETFs (SPY QQQ IWM XLK XLF XLE XLV), 1m RTH.
#     forex    : 8 FX pairs (EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD
#                EURGBP), 1m tick-bar parquet with proxy volume = trade count.
#     -> 27 instruments total.
#
#   Per instrument: matched DOLLAR bars (~6000), causal close-to-close log
#   returns, then backward-only features:
#     CUSUM event sampler, SADF explosiveness (rolling capped backward window;
#     Numba parallel hot loop), rolling Shannon / Lempel-Ziv / Kontoyiannis
#     entropy of the binary return string. Honest costed test: fixed-horizon
#     label, per-feature sign rule fit in-fold, PURGED K-fold CV + embargo,
#     headline = Deflated Sharpe Ratio (lib/overfit) + CSCV PBO across the
#     feature corpus. One descriptive regime figure (no lookahead).
#
#   ESTIMATED RUNTIME : ~5-9 min wall on this 32-core box. The SADF kernel is
#                       Numba parallel=True (uses all cores by default); the
#                       rest is light. Dominant cost is loading the 27 large
#                       1m parquet/csv.gz files (~1.5 GB read).  Single-core
#                       worst case ~3-4 min of pure kernel time.
#   ESTIMATED PEAK RAM: < 2 GB. Instruments are processed ONE AT A TIME and the
#                       base series is released before the next load; the
#                       largest transient is a single 1m crypto parquet
#                       (~1.5M rows -> ~0.9 GB RSS observed). Well under the
#                       46 GB box; no tiling needed at this universe size.
#   OUTPUTS:
#     tables/feature_oos_full.csv             per (instrument,feature) OOS stats
#     tables/feature_sharpe_by_market_full.csv  mean OOS Sharpe pivot
#     tables/headline_full.md                 DSR / SR0 / PBO headline
#     figures/fig1_features_across_regimes.png descriptive regime panel
#
#   Kernels are verified bit-identical vs numpy/python references
#   (run_breaks_entropy.py --verify): CUSUM 0, SADF ~1e-13 (OLS round-off),
#   capped-SADF ~9e-13, Kontoyiannis match-length 0, LZ76 0.
#
#   IDEMPOTENT: overwrites its own tables/figures; re-runnable.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/scripts"

# Pin BLAS/MKL to 1 thread per process so Numba's parallel SADF loop owns the
# cores (avoids oversubscription). Leave NUMBA_NUM_THREADS unset = all cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Sanity: confirm kernels are bit-identical to references before the full run.
python3 run_breaks_entropy.py --verify

# Full multi-market run (fixed-horizon label).
python3 run_breaks_entropy.py --label fixed
