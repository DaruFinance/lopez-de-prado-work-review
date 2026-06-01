#!/usr/bin/env bash
# =============================================================================
# run_full.sh — Project 12: Optimal Trading Rules without backtesting (OU) +
#               Triple Penance, full multi-market run.
#
# López de Prado, AFML Ch.13; Bailey & López de Prado, "Determining Optimal
# Trading Rules without Backtesting" (2014) and "Stop-Outs Under Serial
# Correlation / the Triple Penance Rule" (2015).
#
# WHAT IT DOES
#   Runs the full 42-instrument universe (27 crypto perps + 7 US-equity ETFs +
#   8 FX pairs) on REAL 1-minute data. Per instrument: builds dollar/tick bars,
#   forms a causal z-scored mean-reversion level series, fits an OU/AR(1) on the
#   IN-SAMPLE slice, derives the optimal (profit-take, stop-loss) rule via a
#   Monte-Carlo mesh on the FITTED OU process (the sanctioned synthetic step —
#   params fit to real causal IS data, no lookahead), then VALIDATES the rule on
#   REAL OUT-OF-SAMPLE bars with full intrabar-OHLC first-touch exits and costs.
#   Headline = Deflated Sharpe Ratio (OU rule vs an IS-tuned fixed PT/SL control),
#   plus PBO / effective-N and the Triple Penance serial-correlation-aware
#   max-drawdown / time-under-water bounds.
#
# COST / DATA NOTES
#   Per-side costs: crypto 7 bp, equities 2 bp, forex 1 bp, charged on entry AND
#   exit (full turnover = 2x). 27 IS-tunable trials/instrument feed the DSR.
#
# RUNTIME ESTIMATE (single core, this box)
#   ~35 s per crypto instrument (dominated by the 27x OU Monte-Carlo meshes:
#   1.07 s each at 12x12 grid x 20,000 paths x horizon 500, Numba). Equities add
#   csv.gz decompression overhead. 42 instruments -> ~30-40 min wall, 1 core.
#   This is an OVERNIGHT-SAFE single-core job; do NOT pin more threads — the MC
#   kernel is already the bottleneck and is deterministic single-thread.
#
# PEAK RAM
#   One instrument in memory at a time (loop, not a list). Peak is driven by the
#   largest input frame: the multi-year equity csv.gz (SPY/QQQ, ~19y 1m) loads to
#   ~2.2 GB resident at decompression; the MC random stream is 20000x500 float64
#   = 80 MB; bar/return arrays are ~20k rows (negligible). Steady-state peak
#   ~= 2.2 GB. Safe with wide margin on a 46 GB box. (Crypto/FX instruments peak
#   far lower, ~0.3-0.4 GB.)
#
# OUTPUTS
#   tables/per_instrument.csv, tables/by_market_summary.csv, tables/results.md,
#   tables/raw_results.parquet, figures/fig{1..4}_*.png (200 dpi).
#
# REPRODUCE (exact command)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Single core, deterministic. The OU MC mesh is the hot loop and is single-thread
# by design; capping BLAS/OMP/Numba threads keeps it reproducible and prevents
# oversubscription on a shared box.
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

python3 scripts/run_optimal_trading_rules.py
