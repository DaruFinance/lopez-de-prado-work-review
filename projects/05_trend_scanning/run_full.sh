#!/usr/bin/env bash
# =============================================================================
# Project 05, Trend-Scanning Labels (LdP, ML4AM Ch.5), FULL-SCALE RUN
# =============================================================================
# WHAT IT DOES
#   Builds LdP trend-scanning labels (sign of the max-|t| forward-OLS-slope, with
#   |t| as confidence/size) and a fixed-horizon control labeller, then runs the
#   leakage-free experiment across Crypto + US Equities + Forex:
#     - LABELS are forward-looking targets (trend-scan vs fixed-horizon).
#     - A bagged-tree SECONDARY model PREDICTS the label sign from CAUSAL features
#       under PURGED K-FOLD CV (embargo); the strategy trades the OUT-OF-FOLD
#       predicted side over a causal forward hold, net of per-turnover costs.
#     - Headline metric = DEFLATED SHARPE RATIO (9-trial IS-tunable grid = trials);
#       also PBO (CSCV) and effective-N. Compares trend-scan vs fixed-horizon.
#
# UNIVERSE (all 3 markets, >=10 per market)
#   Crypto   : 27 Binance USD-M perp 1m -> dollar bars (~20k/inst)
#   Equities :  7 Algoseek sector/index ETF 1m (RTH) -> dollar bars
#   Forex    :  8 HistData majors 1m -> tick bars (FX has no volume)
#   = 42 instruments.
#
# COSTS (per side, applied entry+exit): crypto 7 bp / equities 2 bp / forex 1 bp.
#
# OUTPUTS (idempotent; overwrites)
#   tables/per_instrument.csv      per-instrument trend-scan vs fixed-horizon
#   tables/by_market_summary.csv   by-market medians + DSR>0.95 counts
#   tables/results.md              markdown summary
#   tables/raw_results.parquet     full raw result frame
#   figures/fig1_pf_dsr_by_market.png
#   figures/fig2_tval_confidence.png
#   figures/fig3_equity_curves.png
#   figures/fig4_dsr_scatter.png
#
# PERFORMANCE  (measured 2026-05-31 on this box, single core)
#   Hot loop = per-observation multi-horizon OLS t-value scan -> Numba @njit.
#     Numba vs pure-NumPy reference: ~1176x  (n=3000, band (20,120): 4502ms->3.8ms)
#     Full 20k-bar scan, widest band (20,120): ~23 ms/call.
#   Per-instrument wall (9 trials, 6-fold purged CV x2 targets): ~25-35 s
#     (dominated by sklearn tree fits in the meta-CV, NOT the kernel; equities
#      slower due to full multi-year csv.gz decompression).
#   EST FULL RUNTIME : ~20-25 min (42 instruments, SERIAL, single core).
#   PEAK RAM         : ~2.0-2.5 GB (heaviest single instrument = full SPY csv.gz;
#                      instruments processed serially & freed; trial matrices are
#                      T_obs x 9, effective-N corr is 9x9 -> NO N x N blowup).
#                      Well under the 46 GB box; no streaming/tiling required.
#
# THREADING: pinned to 1 core (BLAS/OMP/Numba) so the box stays responsive and
#   results are deterministic; n_jobs=1 in every estimator. The job is
#   embarrassingly parallel ACROSS instruments if a future run wants a pool.
#
# VERIFY (kernel vs independent reference) before the heavy run:
#   python3 scripts/run_trend_scanning.py --verify
#     -> label & lstar bit-identical; t_val max|d| ~1e-6 (float summation order
#        only; realised trade return bit-identical at 0.0 -> P&L unaffected).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

LOG="full_run.log"
echo "=== Project 05 trend-scanning FULL run start: $(date -u) ===" | tee "$LOG"

# kernel verification gate (fast)
python3 scripts/run_trend_scanning.py --verify 2>&1 | tee -a "$LOG"

# full-scale run
python3 scripts/run_trend_scanning.py 2>&1 | tee -a "$LOG"

echo "=== done: $(date -u) ===" | tee -a "$LOG"
