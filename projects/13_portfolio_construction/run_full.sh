#!/usr/bin/env bash
# =============================================================================
# Project 13 — Portfolio Construction (denoising/detoning, HRP, NCO, TIC vs the
# Markowitz curse).  FULL walk-forward run across BOTH universes and all markets.
#
# LdP sources : AFML Ch.16 (HRP); ML4AM Ch.2/4/7 (MP denoising, detoning, NCO);
#               LdP&Lewis 2018 (TIC); headline overfitting via lib/overfit.py (DSR).
#
# METHODOLOGY : real data only; multi-market (Crypto + US Equities + Forex);
#               causal-only; WALK-FORWARD (weights estimated IN-SAMPLE on a 252-day
#               window, HELD over the following 63-day OOS window, rolled by 63;
#               weights are NEVER scored on the data that built them).
#               Headline = realised OOS portfolio variance (annualised vol) + DSR,
#               plus concentration (HHI / effective-N) and cov condition number.
#
# PERFORMANCE : profiled on 1 core (see header of scripts/run_portfolio.py and the
#               writeup). ~98% of wall time is DATA LOADING (ETF gzip-CSV + parquet),
#               NOT the portfolio math — numpy/scipy eigendecomposition + sklearn
#               linkage are already C/Fortran on a bounded (<=300-asset) universe.
#               The one pure-Python hot spot, HRP recursive bisection, is a Numba
#               njit kernel verified BIT-IDENTICAL to the numpy reference
#               (max|Δw| <= 5.6e-17; run `--selftest`). The assembled asset daily-
#               return panel is cached to data_cache/p13/ so reruns are instant.
#
# EST RUNTIME : assets universe ~50s first build (then instant from cache) + WFO;
#               strategies universe ~34s/market x ~25 markets ~= 14-15 min.
#               TOTAL ~= 15-18 min wall on 1 core. (Spec: safe to leave overnight.)
# EST RAM     : sequential per-market; peak RSS ~2.6 GB (one-time pyarrow read
#               overhead), freed between markets. Covariance is N x N with N<=300
#               (~0.7 MB). No N x N beyond the bounded universe; nothing tiled.
#
# Idempotent: re-running overwrites tables/*.csv and reuses the asset cache.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/scripts"

# Pin to 1 core for reproducibility (the math is tiny; threading only adds noise).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

LOG=../../../data_cache/p13_run_full.log
mkdir -p ../../../data_cache
{
  echo "=== Project 13 full run $(date -u +%FT%TZ) ==="

  # 0) Numba bit-identity gate (must pass before any results are trusted).
  python3 run_portfolio.py --selftest

  # 1) ACROSS ASSETS — ~40 instruments, daily close-to-close, TIC enabled
  #    (one-level taxonomy: crypto majors/alts, equity broad/sector/vol, FX usd/cross).
  python3 run_portfolio.py --universe assets \
      --is-win 252 --oos-win 63 --step 63 --tag assets_full

  # 2) ACROSS STRATEGIES — the LdP use-case: 300 de-correlated strategies / market,
  #    all markets, full available daily-PnL history. (TIC auto-disabled: no taxonomy.)
  python3 run_portfolio.py --universe strategies --n-strat 300 \
      --is-win 252 --oos-win 63 --step 63 --tag strategies_full

  echo "=== done $(date -u +%FT%TZ) ==="
} 2>&1 | tee "$LOG"
